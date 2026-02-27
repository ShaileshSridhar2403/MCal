#!/usr/bin/env python3
"""
Fair Comparison with MULTIPLE random ablations per fraction — CTG (3-class).
Mirrors physionet_multi_ablation.py but adapted for CTG's 3-class setup.
"""

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import xgboost as xgb

np.random.seed(42); torch.manual_seed(42)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"

N_ABLATIONS_TRAIN = 100
N_ABLATIONS_EVAL  = 400

NUM_CLASSES = 3

# ─── Data ────────────────────────────────────────────────────────────────────
features = pd.read_csv(DATA_DIR / "ctg_features.csv")
targets  = pd.read_csv(DATA_DIR / "ctg_targets.csv")
df = pd.concat([features, targets["NSP"]], axis=1)

# Convert NSP from (1,2,3) to (0,1,2)
df["NSP"] = df["NSP"] - 1

# Balance by undersampling to minority class
min_count = df["NSP"].value_counts().min()
balanced_dfs = []
for cls in sorted(df["NSP"].unique()):
    cls_df = df[df["NSP"] == cls]
    if len(cls_df) > min_count:
        cls_df = resample(cls_df, replace=False, n_samples=min_count, random_state=42)
    balanced_dfs.append(cls_df)
df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

TARGET = "NSP"
X_all, y_all = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
for d in [X_train, X_test, y_train, y_test]: d.reset_index(drop=True, inplace=True)

imputer = SimpleImputer(strategy="mean")
X_train_clean = imputer.fit_transform(X_train)
X_test_clean  = imputer.transform(X_test)

n_train, n_test, n_feat = len(X_train), len(X_test), X_train_clean.shape[1]
print(f"CTG Dataset — Train: {n_train}, Test: {n_test}, Features: {n_feat}, Classes: {NUM_CLASSES}")
print(f"Class distribution (train): {np.bincount(y_train.values).tolist()}")
print(f"Class distribution (test):  {np.bincount(y_test.values).tolist()}")
print(f"Ablations per fraction: train={N_ABLATIONS_TRAIN}, eval={N_ABLATIONS_EVAL}")

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

def probs(m, X):
    return torch.from_numpy(m.predict_proba(X).astype(np.float32))

def argmax_dist(p):
    oh = torch.zeros_like(p)
    oh[torch.arange(len(p)), p.argmax(1)] = 1.
    return oh.mean(0)

def kl(q, p, eps=1e-10):
    q = q.clamp(eps); p = p.clamp(eps)
    return (q * (q.log() - p.log())).sum().item()

class MCal(nn.Module):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.head = nn.Linear(nc, nc)
    def forward(self, p):
        return F.softmax(self.head(torch.log(p.clamp(min=1e-8))), dim=1)
    def fit(self, abl_probs, targets, max_steps=3000, lr=1e-3):
        opt = optim.Adam(self.parameters(), lr=lr)
        for _ in range(max_steps):
            opt.zero_grad()
            logits = self.head(torch.log(abl_probs.clamp(min=1e-8)))
            loss = F.cross_entropy(logits, targets)
            if torch.isnan(loss): break
            loss.backward(); nn.utils.clip_grad_norm_(self.parameters(), 1.0); opt.step()

FRACS = [round(i/10, 1) for i in range(1, 10)]

# ─── XGBoost models ─────────────────────────────────────────────────────────
print("\nTraining XGBoost models...")
vanilla = xgb.XGBClassifier(objective="multi:softprob", num_class=NUM_CLASSES,
    eval_metric="mlogloss", eta=0.1, max_depth=6, seed=42, tree_method="hist",
    n_estimators=100, verbosity=0)
vanilla.fit(X_train_clean, y_train)

X_rt = ablate_binomial(X_train_clean, 0.5, seed=0)
retrain = xgb.XGBClassifier(objective="multi:softprob", num_class=NUM_CLASSES,
    eval_metric="mlogloss", eta=0.1, max_depth=6, seed=42, tree_method="hist",
    n_estimators=100, verbosity=0)
retrain.fit(imputer.transform(X_rt), y_train)

va_acc = (vanilla.predict(X_test_clean) == y_test).mean()
rt_acc = (retrain.predict(X_test_clean) == y_test).mean()
print(f"  Vanilla test acc: {va_acc:.4f},  Retrain test acc: {rt_acc:.4f}")

# ─── Build multi-ablation training data per fraction ─────────────────────────
print(f"\nBuilding training data with {N_ABLATIONS_TRAIN} ablations per fraction...")

# Target labels (replicated N_ABLATIONS_TRAIN times)
clean_train_probs = probs(vanilla, X_train_clean)
target_labels_single = clean_train_probs.argmax(dim=1)
target_labels_multi = target_labels_single.repeat(N_ABLATIONS_TRAIN)
print(f"  Target labels shape: {target_labels_multi.shape}")

# Unconditioned: N_ABLATIONS_TRAIN independent binomial ablations concatenated
print("\nTraining Unconditioned MCal...")
uncond_probs_list = []
for a in range(N_ABLATIONS_TRAIN):
    X_abl = ablate_binomial(X_train_clean, 0.5, seed=100 + a)
    uncond_probs_list.append(probs(vanilla, imputer.transform(X_abl)))
uncond_train_probs = torch.cat(uncond_probs_list, dim=0)
print(f"  Uncond training data shape: {uncond_train_probs.shape}")

torch.manual_seed(42)
uncond = MCal()
uncond.fit(uncond_train_probs, target_labels_multi)
print("  Done.")

# Conditioned: N_ABLATIONS_TRAIN independent fraction ablations per frac
cond = {}
print("\nTraining Conditioned MCal calibrators...")
for frac in FRACS:
    cond_probs_list = []
    for a in range(N_ABLATIONS_TRAIN):
        seed = int(frac * 1000) + a
        X_abl = ablate_fraction(X_train_clean, frac, seed=seed)
        cond_probs_list.append(probs(vanilla, imputer.transform(X_abl)))
    cond_train_probs = torch.cat(cond_probs_list, dim=0)

    torch.manual_seed(42)
    cal = MCal()
    cal.fit(cond_train_probs, target_labels_multi)
    cond[frac] = cal
    print(f"  p={frac:.1f} done  (train shape: {cond_train_probs.shape})")

# ─── Evaluate: concatenate N_ABLATIONS_EVAL ablations into one pool ──────────
print(f"\nEvaluating on test set ({N_ABLATIONS_EVAL} ablations x {n_test} = "
      f"{N_ABLATIONS_EVAL * n_test} samples per fraction, concatenated)...")

# Clean reference: replicate clean test probs to match pooled size
clean_test_probs_single = probs(vanilla, X_test_clean)
clean_test_probs_pool = clean_test_probs_single.repeat(N_ABLATIONS_EVAL, 1)
clean_test_dist = argmax_dist(clean_test_probs_pool)

retrain_clean_single = probs(retrain, X_test_clean)
retrain_clean_pool = retrain_clean_single.repeat(N_ABLATIONS_EVAL, 1)
retrain_clean_dist = argmax_dist(retrain_clean_pool)

results = {m: {} for m in ['raw', 'retrain', 'uncond', 'cond']}

for frac in FRACS:
    raw_list, rt_list = [], []
    for a in range(N_ABLATIONS_EVAL):
        seed = 5000 + a * 100 + int(frac * 10)
        X_te_abl = ablate_fraction(X_test_clean, frac, seed=seed)
        X_te_imp = imputer.transform(X_te_abl)
        raw_list.append(probs(vanilla, X_te_imp))
        rt_list.append(probs(retrain, X_te_imp))

    raw_pool = torch.cat(raw_list, dim=0)
    rt_pool  = torch.cat(rt_list, dim=0)

    with torch.no_grad():
        uncond_pool = uncond(raw_pool)
        cond_pool   = cond[frac](raw_pool)

    results['raw'][frac]     = kl(argmax_dist(raw_pool),    clean_test_dist)
    results['retrain'][frac] = kl(argmax_dist(rt_pool),     retrain_clean_dist)
    results['uncond'][frac]  = kl(argmax_dist(uncond_pool), clean_test_dist)
    results['cond'][frac]    = kl(argmax_dist(cond_pool),   clean_test_dist)

    print(f"  p={frac:.1f}  Raw={results['raw'][frac]:.4e}  Retrain={results['retrain'][frac]:.4e}  "
          f"Uncond={results['uncond'][frac]:.4e}  Cond={results['cond'][frac]:.4e}")

# ─── Print results ───────────────────────────────────────────────────────────
eval_size = N_ABLATIONS_EVAL * n_test
train_size = N_ABLATIONS_TRAIN * n_train
print("\n" + "=" * 90)
print(f"RESULTS — CTG (3-class) (train: {train_size}, eval: {eval_size} per fraction, concatenated)")
print("=" * 90)

print(f"\n{'p':>4}  {'Raw':>12}  {'Retrain':>12}  {'Uncond':>12}  {'Cond':>12}  {'Winner':>10}")
print("-" * 70)
for frac in FRACS:
    winner = "COND" if results['cond'][frac] < results['uncond'][frac] else "UNCOND"
    print(f"{frac:>4.1f}  {results['raw'][frac]:>12.4e}  {results['retrain'][frac]:>12.4e}  "
          f"{results['uncond'][frac]:>12.4e}  {results['cond'][frac]:>12.4e}  {winner:>10}")

# ─── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for method, label, color, marker in [
    ('retrain',  'Retrain',          'tab:red',    'o'),
    ('uncond',   'MCal (Uncond.)',    'tab:orange', 's'),
    ('cond',     'MCal (Cond.)',      'tab:blue',   '^'),
]:
    vals = [results[method][f] for f in FRACS]
    ax.plot(FRACS, vals, marker=marker, linewidth=2, label=label, color=color)

ax.set_xlabel("Ablation Fraction", fontsize=12)
ax.set_ylabel("Missingness Bias  $D_{KL}$(ablated $\\|$ clean)", fontsize=12)
ax.set_title(f"Fair Comparison: CTG (3-class)\n"
             f"(train: {train_size}, eval: {eval_size} per fraction, concatenated)", fontsize=11)
ax.set_yscale("log")
ax.set_xticks(FRACS)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "ctg_multi_ablation.png", dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {SCRIPT_DIR / 'ctg_multi_ablation.png'}")
