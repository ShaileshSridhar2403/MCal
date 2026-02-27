#!/usr/bin/env python3
"""
Print argmax class distribution tables for all three tabular datasets.
Uses fewer ablations since we just need stable distributions.
"""

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import xgboost as xgb

np.random.seed(42); torch.manual_seed(42)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"

N_ABLATIONS_TRAIN = 50
N_ABLATIONS_EVAL  = 200

FRACS = [round(i/10, 1) for i in range(1, 10)]

# ─── Shared helpers ──────────────────────────────────────────────────────────
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

class MCal(nn.Module):
    def __init__(self, nc):
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


def run_dataset(name, X_train_clean, X_test_clean, y_train, y_test, imputer, nc, xgb_params):
    n_train, n_test = len(y_train), len(y_test)
    print(f"\n{'#' * 90}")
    print(f"# {name} — {nc} classes, {X_train_clean.shape[1]} features, "
          f"train={n_train}, test={n_test}")
    print(f"{'#' * 90}")

    # Train models
    vanilla = xgb.XGBClassifier(**xgb_params, verbosity=0)
    vanilla.fit(X_train_clean, y_train)

    X_rt = ablate_binomial(X_train_clean, 0.5, seed=0)
    retrain = xgb.XGBClassifier(**xgb_params, verbosity=0)
    retrain.fit(imputer.transform(X_rt), y_train)

    # Train MCal calibrators
    clean_train_probs = get_probs(vanilla, X_train_clean)
    target_labels_single = clean_train_probs.argmax(dim=1)
    target_labels_multi = target_labels_single.repeat(N_ABLATIONS_TRAIN)

    # Unconditioned
    uncond_probs_list = []
    for a in range(N_ABLATIONS_TRAIN):
        X_abl = ablate_binomial(X_train_clean, 0.5, seed=100 + a)
        uncond_probs_list.append(get_probs(vanilla, imputer.transform(X_abl)))
    torch.manual_seed(42)
    uncond = MCal(nc)
    uncond.fit(torch.cat(uncond_probs_list, dim=0), target_labels_multi)

    # Conditioned
    cond = {}
    for frac in FRACS:
        cond_probs_list = []
        for a in range(N_ABLATIONS_TRAIN):
            X_abl = ablate_fraction(X_train_clean, frac, seed=int(frac * 1000) + a)
            cond_probs_list.append(get_probs(vanilla, imputer.transform(X_abl)))
        torch.manual_seed(42)
        cal = MCal(nc)
        cal.fit(torch.cat(cond_probs_list, dim=0), target_labels_multi)
        cond[frac] = cal

    # Clean references
    clean_test_dist = argmax_dist(get_probs(vanilla, X_test_clean))
    retrain_clean_dist = argmax_dist(get_probs(retrain, X_test_clean))

    # Format distribution as string
    def fmt_dist(d):
        return "[" + ", ".join(f"{v:.4f}" for v in d.tolist()) + "]"

    # Print header
    print(f"\nClean reference (vanilla):  {fmt_dist(clean_test_dist)}")
    print(f"Clean reference (retrain):  {fmt_dist(retrain_clean_dist)}")

    cls_header = "  ".join(f"c{i}" for i in range(nc))
    col_w = 6 * nc + 2  # approximate width per distribution column

    print(f"\n{'p':>4}  |  {'Raw':<{col_w}}  |  {'Retrain':<{col_w}}  |  "
          f"{'Uncond':<{col_w}}  |  {'Cond':<{col_w}}  |  {'KL_raw':>8}  {'KL_rt':>8}  "
          f"{'KL_unc':>8}  {'KL_cond':>8}")
    print("-" * (4 + 4 * (col_w + 5) + 4 * 10 + 10))

    for frac in FRACS:
        raw_list, rt_list = [], []
        for a in range(N_ABLATIONS_EVAL):
            seed = 5000 + a * 100 + int(frac * 10)
            X_te_abl = ablate_fraction(X_test_clean, frac, seed=seed)
            X_te_imp = imputer.transform(X_te_abl)
            raw_list.append(get_probs(vanilla, X_te_imp))
            rt_list.append(get_probs(retrain, X_te_imp))

        raw_pool = torch.cat(raw_list, dim=0)
        rt_pool  = torch.cat(rt_list, dim=0)

        with torch.no_grad():
            uncond_pool = uncond(raw_pool)
            cond_pool   = cond[frac](raw_pool)

        d_raw = argmax_dist(raw_pool)
        d_rt  = argmax_dist(rt_pool)
        d_unc = argmax_dist(uncond_pool)
        d_cnd = argmax_dist(cond_pool)

        kl_raw = kl(d_raw, clean_test_dist)
        kl_rt  = kl(d_rt,  retrain_clean_dist)
        kl_unc = kl(d_unc, clean_test_dist)
        kl_cnd = kl(d_cnd, clean_test_dist)

        print(f"{frac:>4.1f}  |  {fmt_dist(d_raw):<{col_w}}  |  {fmt_dist(d_rt):<{col_w}}  |  "
              f"{fmt_dist(d_unc):<{col_w}}  |  {fmt_dist(d_cnd):<{col_w}}  |  "
              f"{kl_raw:>8.2e}  {kl_rt:>8.2e}  {kl_unc:>8.2e}  {kl_cnd:>8.2e}")


# ═════════════════════════════════════════════════════════════════════════════
# DATASET 1: PhysioNet
# ═════════════════════════════════════════════════════════════════════════════
print("Loading PhysioNet...")
df_pn = pd.read_csv(SCRIPT_DIR / "balanced_physionet_dataset_0-30.csv")
X_all, y_all = df_pn.drop(columns=["In-hospital_death"]), df_pn["In-hospital_death"]
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
for d in [X_tr, X_te, y_tr, y_te]: d.reset_index(drop=True, inplace=True)
imp_pn = SimpleImputer(strategy="mean")
X_tr_c = imp_pn.fit_transform(X_tr); X_te_c = imp_pn.transform(X_te)
run_dataset("PhysioNet (binary)", X_tr_c, X_te_c, y_tr, y_te, imp_pn, 2,
            dict(objective="binary:logistic", eval_metric="logloss",
                 eta=0.1, max_depth=6, seed=42, tree_method="hist", n_estimators=100))

# ═════════════════════════════════════════════════════════════════════════════
# DATASET 2: CTG
# ═════════════════════════════════════════════════════════════════════════════
print("\n\nLoading CTG...")
features = pd.read_csv(DATA_DIR / "ctg_features.csv")
targets  = pd.read_csv(DATA_DIR / "ctg_targets.csv")
df_ctg = pd.concat([features, targets["NSP"]], axis=1)
df_ctg["NSP"] = df_ctg["NSP"] - 1
min_c = df_ctg["NSP"].value_counts().min()
bal = []
for cls in sorted(df_ctg["NSP"].unique()):
    c = df_ctg[df_ctg["NSP"] == cls]
    if len(c) > min_c: c = resample(c, replace=False, n_samples=min_c, random_state=42)
    bal.append(c)
df_ctg = pd.concat(bal, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
X_all, y_all = df_ctg.drop(columns=["NSP"]), df_ctg["NSP"]
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
for d in [X_tr, X_te, y_tr, y_te]: d.reset_index(drop=True, inplace=True)
imp_ctg = SimpleImputer(strategy="mean")
X_tr_c = imp_ctg.fit_transform(X_tr); X_te_c = imp_ctg.transform(X_te)
run_dataset("CTG (3-class)", X_tr_c, X_te_c, y_tr, y_te, imp_ctg, 3,
            dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                 eta=0.1, max_depth=6, seed=42, tree_method="hist", n_estimators=100))

# ═════════════════════════════════════════════════════════════════════════════
# DATASET 3: Breast Cancer
# ═════════════════════════════════════════════════════════════════════════════
print("\n\nLoading Breast Cancer...")
bc = load_breast_cancer()
X_df = pd.DataFrame(bc.data, columns=bc.feature_names)
y_s = pd.Series(bc.target, name="diagnosis")
combined = pd.concat([X_df, y_s], axis=1)
min_c = y_s.value_counts().min()
bal = []
for cls in sorted(y_s.unique()):
    c = combined[combined["diagnosis"] == cls]
    if len(c) > min_c: c = resample(c, replace=False, n_samples=min_c, random_state=42)
    bal.append(c)
df_bc = pd.concat(bal, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
X_all, y_all = df_bc.drop(columns=["diagnosis"]), df_bc["diagnosis"]
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
for d in [X_tr, X_te, y_tr, y_te]: d.reset_index(drop=True, inplace=True)
imp_bc = SimpleImputer(strategy="mean")
X_tr_c = imp_bc.fit_transform(X_tr); X_te_c = imp_bc.transform(X_te)
run_dataset("Breast Cancer (binary)", X_tr_c, X_te_c, y_tr, y_te, imp_bc, 2,
            dict(objective="binary:logistic", eval_metric="logloss",
                 eta=0.1, max_depth=6, seed=42, tree_method="hist", n_estimators=100))

print("\n\nDone.")
