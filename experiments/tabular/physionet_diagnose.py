#!/usr/bin/env python3
"""
Diagnostics: why is MCal (Cond.) worse than MCal (Uncond.) in many cases?
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb

np.random.seed(42)
torch.manual_seed(42)

SCRIPT_DIR = Path(__file__).parent

# ─── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(SCRIPT_DIR / "balanced_physionet_dataset_0-30.csv")
TARGET = "In-hospital_death"
X_all, y_all = df.drop(columns=[TARGET]), df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)
for d in [X_train, X_test, y_train, y_test]:
    d.reset_index(drop=True, inplace=True)

imputer = SimpleImputer(strategy="mean")
X_train_clean = imputer.fit_transform(X_train)
X_test_clean  = imputer.transform(X_test)

# ─── Ablation helpers ────────────────────────────────────────────────────────
def ablate_binomial(X, p=0.5, seed=None):
    rng = np.random.default_rng(seed)
    X_a = X.copy().astype(float)
    X_a[rng.random(X_a.shape) < p] = np.nan
    return X_a

def ablate_fraction(X, fraction, seed=None):
    rng = np.random.default_rng(seed)
    X_a = X.copy().astype(float)
    n_remove = int(X_a.shape[1] * fraction)
    if n_remove > 0:
        for i in range(len(X_a)):
            X_a[i, rng.choice(X_a.shape[1], size=n_remove, replace=False)] = np.nan
    return X_a

# ─── Model ───────────────────────────────────────────────────────────────────
def train_xgboost(X, y):
    m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                           eta=0.1, max_depth=6, seed=42, tree_method="hist",
                           n_estimators=100, verbosity=0)
    m.fit(X, y); return m

def probs(model, X):
    return torch.from_numpy(model.predict_proba(X).astype(np.float32))

vanilla_model = train_xgboost(X_train_clean, y_train)
vanilla_clean_train_probs = probs(vanilla_model, X_train_clean)
target_labels_train = vanilla_clean_train_probs.argmax(dim=1)
vanilla_clean_test_probs = probs(vanilla_model, X_test_clean)

# ─── MCal with full loss tracking ────────────────────────────────────────────
class MCalCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)

    def forward(self, p):
        return F.softmax(self.head(torch.log(p.clamp(min=1e-8))), dim=1)

    def fit(self, ablated_probs, target_labels, max_steps=3000, lr=1e-3):
        opt = optim.Adam(self.parameters(), lr=lr)
        losses, accs = [], []
        for _ in range(max_steps):
            opt.zero_grad()
            logits = self.head(torch.log(ablated_probs.clamp(min=1e-8)))
            loss = F.cross_entropy(logits, target_labels)
            if torch.isnan(loss): break
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                acc = (logits.argmax(1) == target_labels).float().mean().item()
            losses.append(loss.item()); accs.append(acc)
        return losses, accs


def argmax_dist(p):
    oh = torch.zeros_like(p)
    oh[torch.arange(len(p)), p.argmax(1)] = 1.
    return oh.mean(0)

def kl(q, p, eps=1e-10):
    q = q.clamp(eps); p = p.clamp(eps)
    return (q * (q.log() - p.log())).sum().item()

ABLATION_FRACTIONS = [round(i/10, 1) for i in range(1, 10)]
vanilla_clean_dist = argmax_dist(vanilla_clean_test_probs)

# ─── DIAG 1: Training convergence ────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIAG 1: Training convergence (final loss, final train-acc)")
print("=" * 70)

# Unconditioned
X_abl_binom = ablate_fraction.__class__ and ablate_binomial(X_train_clean, 0.5, seed=1)
van_abl_binom_probs = probs(vanilla_model, imputer.transform(X_abl_binom))
mcal_uncond = MCalCalibrator()
losses_uncond, accs_uncond = mcal_uncond.fit(van_abl_binom_probs, target_labels_train)
print(f"\nUnconditioned (trained at binom p=0.5):")
print(f"  Final train loss: {losses_uncond[-1]:.6f}  |  Final train acc: {accs_uncond[-1]:.4f}")
print(f"  W =\n{mcal_uncond.head.weight.data.numpy()}")
print(f"  b = {mcal_uncond.head.bias.data.numpy()}")

mcal_cond = {}
for frac in ABLATION_FRACTIONS:
    X_abl_frac = ablate_fraction(X_train_clean, frac, seed=int(frac*100))
    van_abl_frac_probs = probs(vanilla_model, imputer.transform(X_abl_frac))
    cal = MCalCalibrator()
    losses_c, accs_c = cal.fit(van_abl_frac_probs, target_labels_train)
    mcal_cond[frac] = cal
    print(f"\nConditioned p={frac:.1f}:")
    print(f"  Final train loss: {losses_c[-1]:.6f}  |  Final train acc: {accs_c[-1]:.4f}")
    print(f"  W =\n{cal.head.weight.data.numpy()}")
    print(f"  b = {cal.head.bias.data.numpy()}")

# ─── DIAG 2: Distribution trace at each test fraction ────────────────────────
print("\n" + "=" * 70)
print("DIAG 2: Full distribution trace on TEST set for each fraction")
print("=" * 70)
print(f"\nClean reference (vanilla on clean test): {vanilla_clean_dist.tolist()}")
print(f"True test labels:                        "
      f"{[y_test.mean(), 1-y_test.mean()]}")

header = f"{'p':>4}  {'RawVan[0]':>10}  {'Uncond[0]':>10}  {'Cond[0]':>10}  "  \
         f"{'KL_uncond':>10}  {'KL_cond':>10}"
print("\n" + header)
print("-" * len(header))

for frac in ABLATION_FRACTIONS:
    X_test_abl = ablate_fraction(X_test_clean, frac, seed=int(frac*100+999))
    X_test_abl_imp = imputer.transform(X_test_abl)
    van_test_probs = probs(vanilla_model, X_test_abl_imp)

    raw_dist = argmax_dist(van_test_probs)

    with torch.no_grad():
        uncond_dist = argmax_dist(mcal_uncond(van_test_probs))
        cond_dist   = argmax_dist(mcal_cond[frac](van_test_probs))

    kl_u = kl(uncond_dist, vanilla_clean_dist)
    kl_c = kl(cond_dist,   vanilla_clean_dist)

    print(f"{frac:>4.1f}  {raw_dist[0].item():>10.4f}  "
          f"{uncond_dist[0].item():>10.4f}  {cond_dist[0].item():>10.4f}  "
          f"{kl_u:>10.4e}  {kl_c:>10.4e}")

# ─── DIAG 3: Same on TRAINING set (check if cond calibrators overfit) ────────
print("\n" + "=" * 70)
print("DIAG 3: Distribution trace on TRAIN set (overfit check)")
print("=" * 70)
vanilla_clean_dist_train = argmax_dist(vanilla_clean_train_probs)
print(f"\nClean reference (vanilla on clean TRAIN): {vanilla_clean_dist_train.tolist()}")

header2 = f"{'p':>4}  {'RawVan[0]':>10}  {'Uncond[0]':>10}  {'Cond[0]':>10}  "  \
          f"{'KL_uncond':>10}  {'KL_cond':>10}"
print("\n" + header2)
print("-" * len(header2))

for frac in ABLATION_FRACTIONS:
    X_tr_abl = ablate_fraction(X_train_clean, frac, seed=int(frac*100))  # same seed as training!
    X_tr_abl_imp = imputer.transform(X_tr_abl)
    van_tr_probs = probs(vanilla_model, X_tr_abl_imp)

    raw_dist = argmax_dist(van_tr_probs)

    with torch.no_grad():
        uncond_dist_tr = argmax_dist(mcal_uncond(van_tr_probs))
        cond_dist_tr   = argmax_dist(mcal_cond[frac](van_tr_probs))

    kl_u = kl(uncond_dist_tr, vanilla_clean_dist_train)
    kl_c = kl(cond_dist_tr,   vanilla_clean_dist_train)

    print(f"{frac:>4.1f}  {raw_dist[0].item():>10.4f}  "
          f"{uncond_dist_tr[0].item():>10.4f}  {cond_dist_tr[0].item():>10.4f}  "
          f"{kl_u:>10.4e}  {kl_c:>10.4e}")

# ─── DIAG 4: Effective log-odds transformation ────────────────────────────────
print("\n" + "=" * 70)
print("DIAG 4: Effective log-odds transformation learned by each calibrator")
print("       (for binary: calibrator maps log-odds s -> effective s')")
print("=" * 70)

def effective_logodds_transform(cal, lo_range=(-4, 4), n=100):
    """Given a log-odds range, compute what the calibrator outputs."""
    lo = torch.linspace(lo_range[0], lo_range[1], n)
    p1 = torch.sigmoid(lo)
    p0 = 1 - p1
    inp = torch.stack([p0, p1], dim=1)
    with torch.no_grad():
        out = cal(inp)
    out_lo = torch.log(out[:, 1]) - torch.log(out[:, 0])
    return lo.numpy(), out_lo.numpy()

lo_in, lo_out_uncond = effective_logodds_transform(mcal_uncond)
fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=True, sharex=True)
axes = axes.flatten()

axes[0].plot(lo_in, lo_out_uncond, 'tab:orange', linewidth=2)
axes[0].plot(lo_in, lo_in, 'k--', alpha=0.4, label='identity')
axes[0].set_title("MCal Uncond\n(trained p=0.5 binom)", fontsize=8)
axes[0].set_xlabel("log-odds in"); axes[0].set_ylabel("log-odds out")
axes[0].legend(fontsize=7)

for i, frac in enumerate(ABLATION_FRACTIONS):
    lo_in_c, lo_out_c = effective_logodds_transform(mcal_cond[frac])
    ax = axes[i + 1]
    ax.plot(lo_in_c, lo_out_c, 'tab:blue', linewidth=2)
    ax.plot(lo_in_c, lo_in_c, 'k--', alpha=0.4)
    ax.set_title(f"MCal Cond p={frac:.1f}", fontsize=8)
    ax.set_xlabel("log-odds in")

plt.suptitle("Log-odds transformation learned by each calibrator", fontsize=11)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "physionet_logodds_transforms.png", dpi=150, bbox_inches="tight")
print("Saved log-odds transform plot.")

# ─── DIAG 5: Input log-odds distributions at each fraction ───────────────────
print("\n" + "=" * 70)
print("DIAG 5: Mean abs log-odds of vanilla predictions on TEST data")
print("        (measures how much signal is left after ablation)")
print("=" * 70)
print(f"\n{'p':>4}  {'mean|log-odds|':>16}  {'std|log-odds|':>14}  {'frac close to 0.5':>18}")
for frac in ABLATION_FRACTIONS:
    X_test_abl = ablate_fraction(X_test_clean, frac, seed=int(frac*100+999))
    X_test_abl_imp = imputer.transform(X_test_abl)
    van_probs = probs(vanilla_model, X_test_abl_imp)
    lo = torch.log(van_probs[:, 1]) - torch.log(van_probs[:, 0])
    mean_abs = lo.abs().mean().item()
    std_abs  = lo.abs().std().item()
    frac_uncertain = (van_probs.max(1).values < 0.55).float().mean().item()
    print(f"{frac:>4.1f}  {mean_abs:>16.4f}  {std_abs:>14.4f}  {frac_uncertain:>18.4f}")

print(f"\n(For reference, clean test:)")
lo_clean = torch.log(vanilla_clean_test_probs[:, 1]) - torch.log(vanilla_clean_test_probs[:, 0])
print(f"  mean|log-odds|={lo_clean.abs().mean().item():.4f}  "
      f"std={lo_clean.abs().std().item():.4f}  "
      f"frac_uncertain={((vanilla_clean_test_probs.max(1).values<0.55).float().mean().item()):.4f}")

print("\nDone.")
