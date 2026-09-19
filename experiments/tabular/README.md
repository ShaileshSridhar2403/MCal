# Tabular benchmarks

Three benchmarks measure missingness bias on tabular data and compare ways
of correcting it:

| Script | Dataset | Classes |
| --- | --- | --- |
| `physionet_kl_benchmark.py` | PhysioNet Challenge 2012, in-hospital mortality | 2 |
| `breast_cancer_kl_benchmark.py` | Wisconsin breast cancer | 2 |
| `ctg_kl_benchmark.py` | Cardiotocography (CTG) | 3 |

Each trains an XGBoost classifier, removes a growing fraction of features
from the test inputs, and measures how far the predicted class distribution
drifts from the clean one (KL divergence), before and after each method.

## Data

The breast cancer data comes with scikit-learn. PhysioNet and CTG need
files under `data/tabular/`; see [docs/datasets.md](../../docs/datasets.md).

## Running

Run from the repository root:

```bash
python -m experiments.tabular.physionet_kl_benchmark \
    --methods baseline replace temperature platt mcal_ce archmod retrain \
    --n_runs 10
```

`run_all_benchmarks.sh` runs all three with these settings.

| Option | Default | Meaning |
| --- | --- | --- |
| `--methods` | `baseline mcal_ce retrain` (PhysioNet adds `mcal_ce_uncond`) | Methods to compare, from the list below |
| `--n_runs` | 3 | Independent runs to average over |
| `--n_samples` | 1000 | Test samples per run |
| `--n_fractions` | 10 | Number of feature-removal fractions |
| `--device` | `cuda` | Use `cpu` without a GPU |
| `--save_dir` | `experiments/tabular/results` | Where results are written |

Methods:

- `baseline`: the classifier's predictions on ablated inputs
- `mcal`, `mcal_ce`: MCal fitted with a KL or cross-entropy loss
  (`mcal_ce_uncond`, PhysioNet only: one calibrator for all fractions)
- `platt`, `temperature`: Platt and temperature scaling
- `logits_sharp`: logit sharpening
- `replace`, `retrain`: alternatives applied when the ablated data and
  model are generated (a different fill value, and a model retrained on
  ablated data)
- `archmod`: architecture-modification baseline

## Outputs

In `--save_dir`:

- `json/aggregated_results_<dataset>.json`: per-method results
- `kl_comparison_table_<dataset>.txt`: summary table
- `kl_divergence_<dataset>.png`: KL divergence against removal fraction

MCal_CE fit summaries, including the learned parameters, go to
`experiments/tabular/results/mcal_ce_combined_results_<dataset>_run_<n>.json`.

## Figures

`plot_physionet_fractionwise_kl.ipynb` draws the PhysioNet panels of
Figures 6 and 7.
