#!/usr/bin/env python3
"""
Verification: is conditioned MCal truly piecewise (right calibrator per fraction)?
And what exactly is the evaluation scheme doing?
"""

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

# ─── Setup (identical to fair_comparison.py) ────────────────────────────────
df = pd.read_csv(SCRIPT_DIR / "balanced_physionet_dataset_0-30.csv")
TARGET = "In-hospital_death"
X_all, y_all = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
for d in [X_train, X_test, y_train, y_test]: d.reset_index(drop=True, inplace=True)

imputer = SimpleImputer(strategy="mean")
X_train_clean = imputer.fit_transform(X_train)
X_test_clean  = imputer.transform(X_test)

def ablate_fraction(X, fraction, seed=None):
    rng = np.random.default_rng(seed)
    X_a = X.copy().astype(float)
    n_remove = int(X_a.shape[1] * fraction)
    if n_remove > 0:
        for i in range(len(X_a)):
            X_a[i, rng.choice(X_a.shape[1], size=n_remove, replace=False)] = np.nan
    return X_a

def ablate_binomial(X, p=0.5, seed=None):
    rng = np.random.default_rng(seed)
    X_a = X.copy().astype(float)
    X_a[rng.random(X_a.shape) < p] = np.nan
    return X_a

def probs(model, X):
    return torch.from_numpy(model.predict_proba(X).astype(np.float32))

vanilla_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                    eta=0.1, max_depth=6, seed=42, tree_method="hist",
                                    n_estimators=100, verbosity=0)
vanilla_model.fit(X_train_clean, y_train)
vanilla_clean_train_probs = probs(vanilla_model, X_train_clean)
target_labels_train = vanilla_clean_train_probs.argmax(dim=1)
vanilla_clean_test_probs = probs(vanilla_model, X_test_clean)

class MCalCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)
    def forward(self, p):
        return F.softmax(self.head(torch.log(p.clamp(min=1e-8))), dim=1)
    def fit(self, ablated_probs, target_labels, max_steps=3000, lr=1e-3):
        opt = optim.Adam(self.parameters(), lr=lr)
        for _ in range(max_steps):
            opt.zero_grad()
            logits = self.head(torch.log(ablated_probs.clamp(min=1e-8)))
            loss = F.cross_entropy(logits, target_labels)
            if torch.isnan(loss): break
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()

def argmax_dist(p):
    oh = torch.zeros_like(p)
    oh[torch.arange(len(p)), p.argmax(1)] = 1.
    return oh.mean(0)

def kl(q, p, eps=1e-10):
    q = q.clamp(eps); p = p.clamp(eps)
    return (q * (q.log() - p.log())).sum().item()

ABLATION_FRACTIONS = [round(i/10, 1) for i in range(1, 10)]
vanilla_clean_dist = argmax_dist(vanilla_clean_test_probs)

# ─── Train calibrators ───────────────────────────────────────────────────────
print("Training calibrators...")

# Unconditioned
X_abl_binom = ablate_binomial(X_train_clean, 0.5, seed=1)
mcal_uncond = MCalCalibrator()
mcal_uncond.fit(probs(vanilla_model, imputer.transform(X_abl_binom)), target_labels_train)

# Conditioned
mcal_cond = {}
for frac in ABLATION_FRACTIONS:
    X_abl = ablate_fraction(X_train_clean, frac, seed=int(frac*100))
    cal = MCalCalibrator()
    cal.fit(probs(vanilla_model, imputer.transform(X_abl)), target_labels_train)
    mcal_cond[frac] = cal

print("Done.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY 1: Confirm piecewise evaluation — show calibrator weights differ per fraction
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("VERIFY 1: Calibrator weights differ per fraction (confirms piecewise)")
print("=" * 70)
print(f"\n{'p':>4}  {'W[0,0]':>8} {'W[0,1]':>8} {'W[1,0]':>8} {'W[1,1]':>8}  {'b[0]':>8} {'b[1]':>8}")
print("-" * 70)
W = mcal_uncond.head.weight.data
b = mcal_uncond.head.bias.data
print(f"{'uncond':>4}  {W[0,0]:>8.4f} {W[0,1]:>8.4f} {W[1,0]:>8.4f} {W[1,1]:>8.4f}  {b[0]:>8.4f} {b[1]:>8.4f}")
for frac in ABLATION_FRACTIONS:
    W = mcal_cond[frac].head.weight.data
    b = mcal_cond[frac].head.bias.data
    print(f"{frac:>4.1f}  {W[0,0]:>8.4f} {W[0,1]:>8.4f} {W[1,0]:>8.4f} {W[1,1]:>8.4f}  {b[0]:>8.4f} {b[1]:>8.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY 2: At p=0.1, show uncond vs cond p=0.1 vs cond p=0.9 applied to same input
#           — wrong calibrator should give clearly different / worse result
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERIFY 2: Cross-apply calibrators (sanity check that they differ)")
print("  All applied to vanilla predictions on 10%-ablated TEST data")
print("=" * 70)

X_test_abl01 = ablate_fraction(X_test_clean, 0.1, seed=1009)
van01 = probs(vanilla_model, imputer.transform(X_test_abl01))

print(f"\n  Raw vanilla dist at p=0.1:                 {argmax_dist(van01).tolist()}")
with torch.no_grad():
    d_uncond = argmax_dist(mcal_uncond(van01))
    d_cond01 = argmax_dist(mcal_cond[0.1](van01))
    d_cond05 = argmax_dist(mcal_cond[0.5](van01))
    d_cond09 = argmax_dist(mcal_cond[0.9](van01))

print(f"  After uncond calibrator  (trained p=0.5 binom): {d_uncond.tolist()}")
print(f"  After cond  calibrator   (trained p=0.1):        {d_cond01.tolist()}")
print(f"  After WRONG cond cal     (trained p=0.5):        {d_cond05.tolist()}")
print(f"  After WRONG cond cal     (trained p=0.9):        {d_cond09.tolist()}")
print(f"  Clean reference (vanilla on clean test):         {vanilla_clean_dist.tolist()}")
print(f"  KL(uncond  || clean): {kl(d_uncond, vanilla_clean_dist):.6f}")
print(f"  KL(cond0.1 || clean): {kl(d_cond01, vanilla_clean_dist):.6f}")
print(f"  KL(cond0.5 || clean): {kl(d_cond05, vanilla_clean_dist):.6f}")
print(f"  KL(cond0.9 || clean): {kl(d_cond09, vanilla_clean_dist):.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY 3: Count per-sample prediction changes at each fraction
#           Shows what calibration actually DOES to individual predictions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERIFY 3: Per-sample prediction flip counts on TEST set")
print("  (how many of 210 test samples change their argmax class after calibration)")
print("=" * 70)

print(f"\n{'p':>4}  {'raw_cls0':>9}  {'uncond_cls0':>12}  {'cond_cls0':>10}  "
      f"{'flips_uncond':>12}  {'flips_cond':>10}  {'clean_ref':>10}")
print("-" * 90)

for frac in ABLATION_FRACTIONS:
    X_test_abl = ablate_fraction(X_test_clean, frac, seed=int(frac*100+999))
    van_p = probs(vanilla_model, imputer.transform(X_test_abl))
    raw_cls = van_p.argmax(1)  # (210,)

    with torch.no_grad():
        uncond_cls = mcal_uncond(van_p).argmax(1)
        cond_cls   = mcal_cond[frac](van_p).argmax(1)

    raw_cls0    = (raw_cls   == 0).sum().item()
    uncond_cls0 = (uncond_cls == 0).sum().item()
    cond_cls0   = (cond_cls   == 0).sum().item()

    flips_uncond = (raw_cls != uncond_cls).sum().item()
    flips_cond   = (raw_cls != cond_cls).sum().item()

    # Also show direction of flips
    raw_0_to_1_uncond = ((raw_cls==0) & (uncond_cls==1)).sum().item()
    raw_1_to_0_uncond = ((raw_cls==1) & (uncond_cls==0)).sum().item()
    raw_0_to_1_cond   = ((raw_cls==0) & (cond_cls==1)).sum().item()
    raw_1_to_0_cond   = ((raw_cls==1) & (cond_cls==0)).sum().item()

    clean_ref_cls0 = int(vanilla_clean_dist[0].item() * 210)

    print(f"{frac:>4.1f}  {raw_cls0:>9}  {uncond_cls0:>12}  {cond_cls0:>10}  "
          f"{flips_uncond:>6}({raw_0_to_1_uncond:+d},{-raw_1_to_0_uncond:+d})  "
          f"{flips_cond:>4}({raw_0_to_1_cond:+d},{-raw_1_to_0_cond:+d})  "
          f"{clean_ref_cls0:>10}")

print(f"\n  Clean ref: {vanilla_clean_dist.tolist()} → ~{clean_ref_cls0} class-0 out of 210")
print(f"  (flip notation: +N means N samples flipped from class0→class1, -N means class1→class0)")

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY 4: Training objective vs. distributional correction
#           Key question: does cross-entropy loss optimize distribution matching?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERIFY 4: CE loss optimizes per-sample accuracy, NOT distribution matching")
print("  Comparing: what distribution does the calibrator achieve on TRAIN data")
print("  vs. what it achieves on TEST data (using the SAME ablation fraction)")
print("=" * 70)

print(f"\n{'p':>4}  {'train_raw_cls0':>15}  {'train_cond_cls0':>16}  "
      f"{'test_raw_cls0':>14}  {'test_cond_cls0':>15}  {'clean_ref_cls0':>15}")
print("-" * 90)

for frac in ABLATION_FRACTIONS:
    # Training data with SAME seed as calibrator was trained on
    X_tr_abl = ablate_fraction(X_train_clean, frac, seed=int(frac*100))
    van_tr = probs(vanilla_model, imputer.transform(X_tr_abl))
    with torch.no_grad():
        cond_tr = mcal_cond[frac](van_tr).argmax(1)
    tr_raw_cls0  = (van_tr.argmax(1) == 0).float().mean().item()
    tr_cond_cls0 = (cond_tr == 0).float().mean().item()

    # Test data
    X_te_abl = ablate_fraction(X_test_clean, frac, seed=int(frac*100+999))
    van_te = probs(vanilla_model, imputer.transform(X_te_abl))
    with torch.no_grad():
        cond_te = mcal_cond[frac](van_te).argmax(1)
    te_raw_cls0  = (van_te.argmax(1) == 0).float().mean().item()
    te_cond_cls0 = (cond_te == 0).float().mean().item()

    clean_ref_cls0 = vanilla_clean_dist[0].item()

    print(f"{frac:>4.1f}  {tr_raw_cls0:>15.4f}  {tr_cond_cls0:>16.4f}  "
          f"{te_raw_cls0:>14.4f}  {te_cond_cls0:>15.4f}  {clean_ref_cls0:>15.4f}")

print(f"\n  Clean ref (train): {argmax_dist(vanilla_clean_train_probs)[0].item():.4f}")
print(f"  Clean ref (test):  {vanilla_clean_dist[0].item():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT: Effective log-odds transformation curve for uncond vs cond p=0.1
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

lo_in = torch.linspace(-8, 8, 200)
p1 = torch.sigmoid(lo_in); p0 = 1 - p1
inp = torch.stack([p0, p1], dim=1)

for ax, (label, cal, color) in zip(
    axes,
    [("MCal Uncond (trained binom p=0.5)", mcal_uncond, "tab:orange"),
     ("MCal Cond p=0.1 (trained frac=0.1)", mcal_cond[0.1], "tab:blue")]
):
    with torch.no_grad():
        out = cal(inp)
    lo_out = torch.log(out[:, 1]) - torch.log(out[:, 0])
    ax.plot(lo_in.numpy(), lo_out.numpy(), color=color, linewidth=2.5, label=label)
    ax.plot(lo_in.numpy(), lo_in.numpy(), 'k--', alpha=0.4, label='identity')
    ax.axhline(0, color='gray', alpha=0.2)
    ax.axvline(0, color='gray', alpha=0.2)
    ax.set_xlabel("Input log-odds (vanilla ablated)", fontsize=11)
    ax.set_ylabel("Output log-odds (calibrated)", fontsize=11)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("Log-odds transformation: what does each calibrator learn?", fontsize=12)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "physionet_logodds_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved log-odds comparison plot.")
