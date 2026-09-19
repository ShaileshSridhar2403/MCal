# MCal: Missingness Bias Calibration

Code for [*Missingness Bias Calibration in Feature Attribution Explanations*](https://arxiv.org/abs/2603.04831)
by Shailesh Sridhar, Anton Xue and Eric Wong.

Explanation methods such as LIME and SHAP probe a model with inputs that
have features removed. Those inputs are out of distribution, and the model's
predictions on them are skewed: missingness bias. MCal corrects it after the
fact by training a small linear head on the outputs of the frozen model,
with no retraining or architecture changes. This repository contains the
`mcal` package and the code to reproduce the paper's experiments on vision
(Brain MRI, CheXpert, BreakHis), language (MedQA, MedMCQA) and tabular
(PhysioNet, breast cancer, cardiotocography) data.

## Install

Requires Python 3.10 or later; developed with Python 3.11.

```bash
git clone https://github.com/ShaileshSridhar2403/MCal.git
cd MCal
pip install -e ".[experiments]"
```

`pip install -e .` installs only the calibrators and their core
dependencies. `requirements-lock.txt` has the exact development pins.

## Using MCal

```python
import torch
from mcal.calibrators import MCal_CE

# A frozen classifier's class probabilities on ablated inputs, and the labels
# it predicts on the same inputs without ablation
ablated_probs = torch.softmax(torch.randn(1000, 4), dim=1)
clean_labels = torch.randint(0, 4, (1000,))

calibrator = MCal_CE(num_classes=4)
calibrator.fit(ablated_probs, clean_labels, max_steps=500)
calibrated_probs = calibrator(ablated_probs)
```

`mcal.calibrators` also provides `MCal` (fitted to a target class
distribution), `TemperatureScaling` and `PlattCalibrator`.

## Reproducing the paper

1. Get the datasets and weights: [docs/datasets.md](docs/datasets.md).
2. Run the benchmarks behind Table 1 and Figures 6 and 7 (GPU, several hours):

   ```bash
   bash run_all_benchmarks.sh
   ```

3. Run the faithfulness experiments:

   ```bash
   python -m experiments.run_faithfulness --dataset mri
   ```

[REPRODUCING.md](REPRODUCING.md) maps every figure and table to the code and
data that produce it, and lists what cannot yet be reproduced.

Experiment scripts are run from the repository root with `python -m`.
Datasets are read from `MCAL_DATA_ROOT` (default `data/`) and weights from
`MCAL_MODEL_ROOT` (default `saved_models/`).

## Repository layout

```
mcal/                 the MCal package: calibrators, transforms, data loading
experiments/          benchmarks, faithfulness experiments and figure notebooks
  vision/ language/ tabular/
  faithfulness/       one faithfulness experiment per dataset
notebooks/            Table 1
examples/             Figure 4 (synthetic data)
vit_patch_drop/       vendored Vision Transformer used by the vision benchmarks
docs/                 datasets and methodology notes
tests/                test suite
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Licence

MIT, see [LICENSE](LICENSE). `vit_patch_drop/` is third-party code under
its own terms; see [vit_patch_drop/NOTICE](vit_patch_drop/NOTICE).

## Citation

```bibtex
@misc{sridhar2026mcal,
  title         = {Missingness Bias Calibration in Feature Attribution Explanations},
  author        = {Sridhar, Shailesh and Xue, Anton and Wong, Eric},
  year          = {2026},
  eprint        = {2603.04831},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2603.04831}
}
```
