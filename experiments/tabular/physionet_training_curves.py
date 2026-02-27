#!/usr/bin/env python3
"""
Training curves: KL divergence and CE loss vs steps for each MCal calibrator.
"""

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb

np.random.seed(42); torch.manual_seed(42)
SCRIPT_DIR = Path(__file__).parent

N_ABLATIONS_TRAIN = 100
N_ABLATIONS_EVAL  = 50   # smaller for speed — just need stable curves
MAX_STEPS = 10000
LR = 1e-2
LOG_EVERY = 50  # log metrics every N steps

# ─── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(SCRIPT_DIR / "balanced_physionet_dataset_0-30.csv")
TARGET = "In-hospital_death"
X_all, y_all = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
for d in [X_train, X_test, y_train, y_test]: d.reset_index(drop=True, inplace=True)

imputer = SimpleImputer(strategy="mean")
X_train_clean = imputer.fit_transform(X_train)
X_test_clean  = imputer.transform(X_test)

n_train, n_test = len(X_train), len(X_test)
print(f"Train: {n_train}, Test: {n_test}, Features: {X_train_clean.shape[1]}")

# ─── Helpers ─────────────────────────────────────────────────────────────────
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

def get_probs(m, X):
    return torch.from_numpy(m.predict_proba(X).astype(np.float32))

def argmax_dist(p):
    oh = torch.zeros_like(p)
    oh[torch.arange(len(p)), p.argmax(1)] = 1.
    return oh.mean(0)

def kl(q, p, eps=1e-10):
    q = q.clamp(eps); p = p.clamp(eps)
    return (q * (q.log() - p.log())).sum().item()

FRACS = [round(i/10, 1) for i in range(1, 10)]

# ─── XGBoost ─────────────────────────────────────────────────────────────────
print("Training XGBoost...")
vanilla = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
    eta=0.1, max_depth=6, seed=42, tree_method="hist", n_estimators=100, verbosity=0)
vanilla.fit(X_train_clean, y_train)

# ─── Clean reference ─────────────────────────────────────────────────────────
clean_test_dist = argmax_dist(get_probs(vanilla, X_test_clean))

# ─── Pre-compute training data ───────────────────────────────────────────────
print("Pre-computing training data...")
clean_train_probs = get_probs(vanilla, X_train_clean)
target_labels_single = clean_train_probs.argmax(dim=1)
target_labels_multi = target_labels_single.repeat(N_ABLATIONS_TRAIN)

# Unconditioned training data
uncond_probs_list = []
for a in range(N_ABLATIONS_TRAIN):
    X_abl = ablate_binomial(X_train_clean, 0.5, seed=100 + a)
    uncond_probs_list.append(get_probs(vanilla, imputer.transform(X_abl)))
uncond_train_probs = torch.cat(uncond_probs_list, dim=0)

# Conditioned training data per fraction
cond_train_data = {}
for frac in FRACS:
    cond_probs_list = []
    for a in range(N_ABLATIONS_TRAIN):
        X_abl = ablate_fraction(X_train_clean, frac, seed=int(frac * 1000) + a)
        cond_probs_list.append(get_probs(vanilla, imputer.transform(X_abl)))
    cond_train_data[frac] = torch.cat(cond_probs_list, dim=0)

# ─── Pre-compute eval pools (smaller, for periodic KL evaluation) ────────────
print("Pre-computing eval pools...")
eval_pools = {}
for frac in FRACS:
    raw_list = []
    for a in range(N_ABLATIONS_EVAL):
        seed = 5000 + a * 100 + int(frac * 10)
        X_te_abl = ablate_fraction(X_test_clean, frac, seed=seed)
        raw_list.append(get_probs(vanilla, imputer.transform(X_te_abl)))
    eval_pools[frac] = torch.cat(raw_list, dim=0)

# Also for uncond: eval at each fraction
# (uncond is applied to every fraction's ablated data)

# ─── Training with logging ───────────────────────────────────────────────────
def train_with_logging(name, train_probs, targets, eval_pool, clean_ref):
    """Train MCal and log CE loss + KL at each fraction every LOG_EVERY steps."""
    torch.manual_seed(42)
    model = nn.Linear(2, 2)
    opt = optim.Adam(model.parameters(), lr=LR)

    steps_log = []
    ce_log = []
    kl_log = []

    for step in range(MAX_STEPS):
        opt.zero_grad()
        logits = model(torch.log(train_probs.clamp(min=1e-8)))
        loss = F.cross_entropy(logits, targets)
        if torch.isnan(loss): break
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            with torch.no_grad():
                # CE loss
                ce_log.append(loss.item())

                # KL on eval pool
                cal_probs = F.softmax(model(torch.log(eval_pool.clamp(min=1e-8))), dim=1)
                cal_dist = argmax_dist(cal_probs)
                kl_val = kl(cal_dist, clean_ref)
                kl_log.append(kl_val)

                steps_log.append(step)

    return steps_log, ce_log, kl_log


# ─── Train unconditioned (evaluate at each fraction) ─────────────────────────
print("\nTraining Unconditioned MCal with logging...")
torch.manual_seed(42)
uncond_model = nn.Linear(2, 2)
uncond_opt = optim.Adam(uncond_model.parameters(), lr=LR)

uncond_steps = []
uncond_ce = []
uncond_kl_per_frac = {frac: [] for frac in FRACS}

for step in range(MAX_STEPS):
    uncond_opt.zero_grad()
    logits = uncond_model(torch.log(uncond_train_probs.clamp(min=1e-8)))
    loss = F.cross_entropy(logits, target_labels_multi)
    if torch.isnan(loss): break
    loss.backward()
    nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
    uncond_opt.step()

    if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
        with torch.no_grad():
            uncond_ce.append(loss.item())
            uncond_steps.append(step)
            for frac in FRACS:
                cal_probs = F.softmax(uncond_model(torch.log(eval_pools[frac].clamp(min=1e-8))), dim=1)
                uncond_kl_per_frac[frac].append(kl(argmax_dist(cal_probs), clean_test_dist))

print("  Done.")

# ─── Train conditioned (one per fraction) ────────────────────────────────────
print("Training Conditioned MCal calibrators with logging...")
cond_steps = {}
cond_ce = {}
cond_kl = {}

for frac in FRACS:
    s, ce, kl_vals = train_with_logging(
        f"Cond p={frac:.1f}",
        cond_train_data[frac], target_labels_multi,
        eval_pools[frac], clean_test_dist
    )
    cond_steps[frac] = s
    cond_ce[frac] = ce
    cond_kl[frac] = kl_vals
    print(f"  p={frac:.1f} done — final CE={ce[-1]:.4f}, final KL={kl_vals[-1]:.4e}")

# ─── Plot 1: CE loss curves ─────────────────────────────────────────────────
print("\nPlotting...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
axes = axes.flatten()

# First panel: unconditioned CE
axes[0].plot(uncond_steps, uncond_ce, color='tab:orange', linewidth=1.5)
axes[0].set_title("Uncond (binom p=0.5)", fontsize=9)
axes[0].set_ylabel("CE Loss", fontsize=9)
axes[0].set_xlabel("Step", fontsize=8)
axes[0].grid(alpha=0.3)

# Remaining panels: conditioned CE per fraction
for i, frac in enumerate(FRACS):
    ax = axes[i + 1]
    ax.plot(cond_steps[frac], cond_ce[frac], color='tab:blue', linewidth=1.5)
    ax.set_title(f"Cond p={frac:.1f}", fontsize=9)
    ax.set_xlabel("Step", fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("CE Loss During Training (PhysioNet)", fontsize=13)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "physionet_ce_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved CE curves to {SCRIPT_DIR / 'physionet_ce_curves.png'}")

# ─── Plot 2: KL curves ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
axes = axes.flatten()

# First panel: unconditioned KL (pick a representative fraction, or show multiple)
ax = axes[0]
for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
    ax.plot(uncond_steps, uncond_kl_per_frac[frac], linewidth=1.2, label=f"eval p={frac}", alpha=0.8)
ax.set_title("Uncond — KL at various eval fracs", fontsize=9)
ax.set_ylabel("KL Divergence", fontsize=9)
ax.set_xlabel("Step", fontsize=8)
ax.set_yscale("log")
ax.legend(fontsize=6, loc='upper right')
ax.grid(alpha=0.3)

# Remaining panels: conditioned KL per fraction (eval at matching fraction)
for i, frac in enumerate(FRACS):
    ax = axes[i + 1]
    # Cond KL (blue)
    ax.plot(cond_steps[frac], cond_kl[frac], color='tab:blue', linewidth=1.5, label='Cond')
    # Uncond KL at same eval fraction (orange, for comparison)
    ax.plot(uncond_steps, uncond_kl_per_frac[frac], color='tab:orange', linewidth=1.2,
            alpha=0.7, linestyle='--', label='Uncond')
    ax.set_title(f"Eval p={frac:.1f}", fontsize=9)
    ax.set_xlabel("Step", fontsize=8)
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

plt.suptitle("KL Divergence During Training (PhysioNet)\n"
             "Blue=Cond (trained & eval at same p), Orange dashed=Uncond (eval at same p)",
             fontsize=12)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "physionet_kl_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved KL curves to {SCRIPT_DIR / 'physionet_kl_curves.png'}")

print("\nDone.")
