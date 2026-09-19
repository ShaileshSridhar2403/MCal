# Datasets and weights

The repository ships no image data and no trained weights. This page says
where to get each dataset, where to put it, and how to check it.

Most data is read from `MCAL_DATA_ROOT` and weights from `MCAL_MODEL_ROOT`.
With an editable install (`pip install -e .`) these default to `data/` and
`saved_models/` at the repository root; set the environment variables to
point elsewhere. A few vision experiments read copies kept under
`experiments/vision/`, noted below.

SHA-256 checksums are of the files used during development.

| Dataset | Used by | Location | Source |
| --- | --- | --- | --- |
| Brain Tumor MRI | MRI benchmark, Figures 5, 6, 7, 8, 11 | `data/Training`, `data/Testing`; also `experiments/vision/data/` | Kaggle |
| CheXpert (small) | CheXpert benchmark | `experiments/vision/CheXpert-v1.0-small/` | Stanford AIMI, registration required |
| BreakHis | BreakHis benchmark | `data/` and `experiments/vision/data/BreakHis/`, see below | pre-split copy, see below |
| PhysioNet 2012 | PhysioNet benchmark, Figures 6, 7, 8 | `data/tabular/missingness_levels/` | PhysioNet |
| Cardiotocography | CTG benchmark | `data/tabular/ctg/ctg_dataset.csv` | UCI |
| Wisconsin breast cancer | breast cancer benchmark | none | scikit-learn, no download |
| MedQA | MedQA benchmark, Figures 6, 7 | `data/language/medqa_dev_balanced.jsonl` | balanced dev split, see below |
| MedMCQA | MedMCQA benchmark | `data/language/medmcqa_dev_balanced.json` | included in this repository |

## Vision

### Brain Tumor MRI

Nickparvar (2021), https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

```bash
python download_mri_dataset.py   # needs the kaggle CLI and ~/.kaggle/kaggle.json
```

The script unpacks `Training/` and `Testing/` into `experiments/vision/data/`,
which is where the faithfulness experiments and the Figure 5, 8 and 11
notebooks read them. The KL benchmarks read the same folders from
`MCAL_DATA_ROOT`, so also copy or link them into `data/`.

Development archive: `brain-tumor-mri-dataset.zip`,
SHA-256 `048b7413ce05bb52627046cfa5cde097b12e865aa80f6110959f547fd6fe4ef1`.

The two development copies were not identical: `data/Training` held 5,712
images, as in the Kaggle archive, and `experiments/vision/data/Training`
held 5,284. Rerunning the faithfulness experiments on a fresh download may
therefore not match the committed numbers exactly.

### CheXpert

CheXpert is released under a research-use agreement that forbids
redistribution. It is never downloaded automatically and must never be
committed.

1. Register and download CheXpert-v1.0-small from the Stanford AIMI
   shared datasets portal, https://stanfordaimi.azurewebsites.net/
2. Unpack it so that `experiments/vision/CheXpert-v1.0-small/` contains
   `train.csv`, `valid.csv`, `train/` and `valid/`.

### BreakHis

The images come from the BreakHis database,
https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

The benchmark uses a pre-split copy, `dataset_v2.zip`, with `train/` and
`test/` folders and one sub-folder per tumour type (8 classes). Where this
split was obtained is not recorded in the repository.

The benchmark reads the archive through two loaders:

- `mcal.data.BreakHisLoader` reads `data/BreakHisTraining/` and
  `data/BreakHisTesting/` (unpacked from `data/dataset_v2.zip`).
- `experiments/vision/breakhis_data_setup.py` reads
  `experiments/vision/data/BreakHis/BreakHisTraining/` and
  `BreakHisTesting/`. If they are missing it unpacks
  `data/BreakHis/dataset_v2.zip` there.

Put the same archive at both `data/dataset_v2.zip` and
`data/BreakHis/dataset_v2.zip`. Development archive SHA-256:
`c04d975dacafa6136e3b7811a62341035080c5e30e6641ec44e07550a4bd1ed0`.

The development copies unpacked from this one archive were not identical:
both test folders held 401 files, but `data/BreakHisTraining` held 696
and `experiments/vision/data/BreakHis/BreakHisTraining` held 1,594.

## Tabular

### PhysioNet Challenge 2012

https://physionet.org/content/challenge-2012/1.0.0/

The benchmark reads per-missingness-level files from
`data/tabular/missingness_levels/`. Build them from the processed set-a
table:

```bash
python -m experiments.tabular.process_physionet_data \
    --input_path data/tabular/PhysionetChallenge2012-set-a.csv.gz
```

With the development copy of `PhysionetChallenge2012-set-a.csv.gz`
(SHA-256 `75298854f9fb7ffbd4b81387f726aebe474d00f5db751627236cb0ad943736e3`)
this regenerates the development files exactly. The script can also start
from the raw `set-a/` text files (`--from_txt`), but that route was not
checked against the development data.

### Cardiotocography (CTG)

UCI Machine Learning Repository, https://doi.org/10.24432/C51S4N

Download `CTG.xls` and convert its "Raw Data" sheet, dropping the blank
first row:

```python
import pandas as pd
pd.read_excel("CTG.xls", sheet_name="Raw Data").iloc[1:].to_csv(
    "data/tabular/ctg/ctg_dataset.csv", index=False)
```

This reproduces the development `ctg_dataset.csv`. Reading `.xls` files
needs the `xlrd` package. Development `CTG.xls` SHA-256:
`d6aa62e82625e59f9d5b05bc8fc52af9ea67bdf2a23f1faca8511d5618fcb421`.

The CTG faithfulness experiment instead reads `experiments/data/ctg_features.csv`
and `ctg_targets.csv`, which are included.

### Wisconsin breast cancer

Loaded with `sklearn.datasets.load_breast_cancer`; nothing to download.

## Language

### MedQA

The benchmark reads a balanced subset of the MedQA development set,
`data/language/medqa_dev_balanced.jsonl`: 1,070 questions in the MedQA
JSON Lines format (`question`, `options`, `answer_idx`, `meta_info`).
MedQA itself is available from https://github.com/jind11/MedQA. How the
balanced subset was drawn is not recorded in the repository.

Development file SHA-256:
`c7956fd2f01f85aaf4039e580f654b7814a49366c49b45bfa330416ab6f095e4`.

### MedMCQA

A balanced MedMCQA development subset (2,056 questions) is included:

```bash
mkdir -p data/language
cp experiments/language/dataset_store/medmcqa_dev_balanced.jsonl data/language/medmcqa_dev_balanced.json
```

MedMCQA: https://medmcqa.github.io/

## Weights

### Vision Transformers

The vision benchmarks and figures load these checkpoints from
`MCAL_MODEL_ROOT`. They are not yet hosted anywhere public.

| File | SHA-256 |
| --- | --- |
| `vit_timm_standard_mri_ps64_35e.pth` | `943b5ad7ef3028213cf3f5e3658b938d7c88a29d10b66e9e725a734b982439a5` |
| `vit_timm_patchcutoutbinom_mri_ps64_55e.pth` | `ad7a02f6116db1d682ea76d7cfc917d8b9ade28097799c9c3554287d10b77093` |
| `vit_timm_vanilla_breakhis_ps56_98tr89te.pth` | `8cfde387f3d6a0c893bc979db963760e2ea39d5e27b14d9de62d3ec1faaf5afb` |
| `vit_timm_PatchCutout_breakhis_ps56_98tr85te.pth` | `4d37871b4cf7f7ed4dd4cf0852d3926c40693763365f883be8265c294e1659db` |
| `vit_timm_vanilla_chexpert_ps64_85tr76te.pth` | `0e9b7c757c95400a94ac06e165dc50a3988c1d9c272797b6cf00145fb6226841` |
| `vit_timm_patchcutout_chexpert_ps64_83tr75te.pth` | `e49989c8bcbfbd522000584b1c4932535294c7df47998fa1b0994e210b91cd7e` |

### Language models

The language benchmarks use Meta-Llama-3-8B-Instruct, which requires
accepting Meta's licence on Hugging Face. Place it at
`MCAL_MODEL_ROOT/language/Meta-Llama-3-8B-Instruct/`.

The `qlora` model variants are fine-tuned from it:

```bash
bash experiments/language/finetune_medical_qa_model.sh --dataset medqa --p_ablate 0.5
```

This writes `MCAL_MODEL_ROOT/medqa/medqa_p0.5/merged_model`, where the
MedQA benchmark looks for it.
