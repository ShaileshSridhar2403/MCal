#!/usr/bin/env python3
"""
Fair Comparison: Retrain vs MCal (Unconditioned & Conditioned) on PhysioNet.
Proper 80/20 train/test split — calibrators fitted on train, evaluated on test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

SCRIPT_DIR = Path(__file__).parent

# ─── 1. DATA ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("1. Loading data")
print("=" * 60)

data_path = SCRIPT_DIR / "balanced_physionet_dataset_0-30.csv"
df = pd.read_csv(data_path)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target distribution:\n{df['In-hospital_death'].value_counts().to_dict()}")

TARGET = "In-hospital_death"
X_all = df.drop(columns=[TARGET])
y_all = df[TARGET]
print(f"Features: {X_all.shape[1]}")

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)
X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

print(f"\nTrain: {len(X_train)} samples  {y_train.value_counts().to_dict()}")
print(f"Test:  {len(X_test)} samples   {y_test.value_counts().to_dict()}")

# Imputer fitted ONLY on clean training data
imputer = SimpleImputer(strategy="mean")
X_train_clean = imputer.fit_transform(X_train)
X_test_clean  = imputer.transform(X_test)
print("Imputer fitted on training data only.")

# ─── 2. ABLATION FUNCTIONS ───────────────────────────────────────────────────

def ablate_binomial(X_np: np.ndarray, p: float = 0.5, seed: int = None) -> np.ndarray:
    """Coin-flip ablation: each feature independently set to NaN with prob p."""
    rng = np.random.default_rng(seed)
    X_abl = X_np.copy().astype(float)
    mask = rng.random(X_abl.shape) < p
    X_abl[mask] = np.nan
    return X_abl


def ablate_fraction(X_np: np.ndarray, fraction: float, seed: int = None) -> np.ndarray:
    """Deterministic-count ablation: exactly floor(n_feat * fraction) removed per row."""
    rng = np.random.default_rng(seed)
    X_abl = X_np.copy().astype(float)
    n_samples, n_feats = X_abl.shape
    n_remove = int(n_feats * fraction)
    if n_remove > 0:
        for i in range(n_samples):
            cols = rng.choice(n_feats, size=n_remove, replace=False)
            X_abl[i, cols] = np.nan
    return X_abl


# ─── 3. XGBOOST MODELS ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. Training XGBoost models")
print("=" * 60)

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    eta=0.1,
    max_depth=6,
    seed=42,
    tree_method="hist",
    enable_categorical=False,
    n_estimators=100,
    verbosity=0,
)


def train_xgboost(X: np.ndarray, y: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    return model


def probs_from_model(model: xgb.XGBClassifier, X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(model.predict_proba(X).astype(np.float32))


# Vanilla XGBoost
print("Training Vanilla XGBoost (clean data)...")
vanilla_model = train_xgboost(X_train_clean, y_train)
va_train = (vanilla_model.predict(X_train_clean) == y_train).mean()
va_test  = (vanilla_model.predict(X_test_clean)  == y_test).mean()
print(f"  Train acc: {va_train:.4f}  |  Test acc: {va_test:.4f}")

# Retrained XGBoost
print("\nTraining Retrained XGBoost (50%-binomial ablated data)...")
X_train_abl_binom = ablate_binomial(X_train_clean, p=0.5, seed=0)
X_train_abl_imputed = imputer.transform(X_train_abl_binom)
retrain_model = train_xgboost(X_train_abl_imputed, y_train)
rt_train = (retrain_model.predict(X_train_abl_imputed) == y_train).mean()
rt_test  = (retrain_model.predict(X_test_clean) == y_test).mean()
print(f"  Train acc (ablated): {rt_train:.4f}  |  Test acc (clean): {rt_test:.4f}")

# ─── 4. MCAL CALIBRATOR ──────────────────────────────────────────────────────

class MCalCalibrator(nn.Module):
    """Linear MCal calibrator: R(p) = softmax(W log p + b)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.head = nn.Linear(num_classes, num_classes)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        logits = torch.log(probs.clamp(min=1e-8))
        return F.softmax(self.head(logits), dim=1)

    def fit(self, ablated_probs: torch.Tensor, target_labels: torch.Tensor,
            max_steps: int = 3000, lr: float = 1e-3, desc: str = ""):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for _ in range(max_steps):
            optimizer.zero_grad()
            logits = self.head(torch.log(ablated_probs.clamp(min=1e-8)))
            loss = F.cross_entropy(logits, target_labels)
            if torch.isnan(loss):
                break
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()


# ─── 5. TRAIN CALIBRATORS ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. Training MCal calibrators (on training data)")
print("=" * 60)

# Target labels = argmax of vanilla clean predictions on training data
vanilla_clean_train_probs = probs_from_model(vanilla_model, X_train_clean)
target_labels_train = vanilla_clean_train_probs.argmax(dim=1)
print(f"Target label dist (train): {torch.bincount(target_labels_train).tolist()}")

# Unconditioned MCal (trained on 50%-binomial ablated predictions)
print("\nTraining Unconditioned MCal (p=0.5 binomial)...")
X_train_abl_binom_mcal = ablate_binomial(X_train_clean, p=0.5, seed=1)
X_train_abl_binom_mcal_imputed = imputer.transform(X_train_abl_binom_mcal)
vanilla_abl_train_probs = probs_from_model(vanilla_model, X_train_abl_binom_mcal_imputed)

mcal_uncond = MCalCalibrator(num_classes=2)
mcal_uncond.fit(vanilla_abl_train_probs, target_labels_train, desc="MCal-Uncond")
print("  Done.")

# Conditioned MCal (one per fraction)
ABLATION_FRACTIONS = [round(p / 10, 1) for p in range(1, 10)]
mcal_cond = {}

print("\nTraining Conditioned MCal calibrators...")
for frac in ABLATION_FRACTIONS:
    X_train_abl_frac = ablate_fraction(X_train_clean, fraction=frac, seed=int(frac * 100))
    X_train_abl_frac_imputed = imputer.transform(X_train_abl_frac)
    vanilla_abl_frac_probs = probs_from_model(vanilla_model, X_train_abl_frac_imputed)
    cal = MCalCalibrator(num_classes=2)
    cal.fit(vanilla_abl_frac_probs, target_labels_train, desc=f"MCal-Cond p={frac:.1f}")
    mcal_cond[frac] = cal
    print(f"  p={frac:.1f} done")

# ─── 6. EVALUATE ON TEST SET ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. Evaluating on test set")
print("=" * 60)


def argmax_dist(probs: torch.Tensor) -> torch.Tensor:
    """Mean one-hot argmax distribution over samples → (C,) vector."""
    one_hot = torch.zeros_like(probs)
    one_hot[torch.arange(len(probs)), probs.argmax(dim=1)] = 1.0
    return one_hot.mean(dim=0)


def kl_divergence(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-10) -> float:
    """D_KL(q || p) where q = ablated dist, p = clean reference dist."""
    q = q.clamp(min=eps)
    p = p.clamp(min=eps)
    return (q * (q.log() - p.log())).sum().item()


# Clean reference distributions on test set
retrain_clean_test_probs = probs_from_model(retrain_model, X_test_clean)
retrain_clean_dist = argmax_dist(retrain_clean_test_probs)

vanilla_clean_test_probs = probs_from_model(vanilla_model, X_test_clean)
vanilla_clean_dist = argmax_dist(vanilla_clean_test_probs)

print(f"Retrain clean dist (test): {retrain_clean_dist.tolist()}")
print(f"Vanilla clean dist (test): {vanilla_clean_dist.tolist()}")

results = {"fractions": [], "retrain": [], "mcal_uncond": [], "mcal_cond": []}

print(f"\n{'p':>5}  {'Retrain':>12}  {'MCal-Uncond':>13}  {'MCal-Cond':>12}")
print("-" * 50)

for frac in ABLATION_FRACTIONS:
    X_test_abl = ablate_fraction(X_test_clean, fraction=frac, seed=int(frac * 100 + 999))
    X_test_abl_imputed = imputer.transform(X_test_abl)

    # Retrain
    rt_probs = probs_from_model(retrain_model, X_test_abl_imputed)
    kl_rt = kl_divergence(argmax_dist(rt_probs), retrain_clean_dist)

    # Vanilla probs (shared base for both MCal variants)
    van_probs = probs_from_model(vanilla_model, X_test_abl_imputed)

    # Unconditioned MCal
    with torch.no_grad():
        uncond_cal_probs = mcal_uncond(van_probs)
    kl_uncond = kl_divergence(argmax_dist(uncond_cal_probs), vanilla_clean_dist)

    # Conditioned MCal
    with torch.no_grad():
        cond_cal_probs = mcal_cond[frac](van_probs)
    kl_cond = kl_divergence(argmax_dist(cond_cal_probs), vanilla_clean_dist)

    results["fractions"].append(frac)
    results["retrain"].append(kl_rt)
    results["mcal_uncond"].append(kl_uncond)
    results["mcal_cond"].append(kl_cond)

    print(f"{frac:>5.1f}  {kl_rt:>12.4e}  {kl_uncond:>13.4e}  {kl_cond:>12.4e}")

# ─── 7. PLOT ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. Plotting")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 5))
fracs = results["fractions"]

ax.plot(fracs, results["retrain"],     marker="o", linewidth=2, label="Retrain",        color="tab:red")
ax.plot(fracs, results["mcal_uncond"], marker="s", linewidth=2, label="MCal (Uncond.)", color="tab:orange")
ax.plot(fracs, results["mcal_cond"],   marker="^", linewidth=2, label="MCal (Cond.)",   color="tab:blue")

ax.set_xlabel("Ablation Fraction", fontsize=12)
ax.set_ylabel("Missingness Bias  $D_{KL}$(ablated $\\|$ clean)", fontsize=12)
ax.set_title("Fair Comparison: PhysioNet\n(calibrators fitted on train, evaluated on held-out test)", fontsize=11)
ax.set_yscale("log")
ax.set_xticks(fracs)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out_path = SCRIPT_DIR / "physionet_fair_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved figure to {out_path}")

# Also save raw numbers
results_df = pd.DataFrame({
    "ablation_fraction": results["fractions"],
    "Retrain":           results["retrain"],
    "MCal_Uncond":       results["mcal_uncond"],
    "MCal_Cond":         results["mcal_cond"],
})
csv_path = SCRIPT_DIR / "physionet_fair_comparison_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"Saved results to {csv_path}")
