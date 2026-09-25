# Datasets

The code in this repository is MIT-licensed (see [LICENSE](LICENSE)). The
datasets are other people's work and remain under their own licences. A few
small derived files are redistributed here under those licences; everything
else you download from the original source.

If you use any of these datasets, please cite the original authors as well as
this work.

## Included in this repository

### Cardiotocography (CTG)

- **Files:** `experiments/data/ctg_features.csv`, `experiments/data/ctg_targets.csv`
- **Source:** UCI Machine Learning Repository, <https://doi.org/10.24432/C51S4N>
- **Licence:** [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Changes:** features and class labels extracted from the original spreadsheet into two CSV files.
- **Cite:** Campos, D. & Bernardes, J. (2000). *Cardiotocography* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51S4N

### MedMCQA

- **File:** `experiments/language/dataset_store/medmcqa_dev_balanced.jsonl`, a 2,056-question subset of the development split
- **Source:** <https://github.com/MedMCQA/MedMCQA>
- **Licence:** MIT, Copyright (c) 2022 MedMCQA. The full licence text accompanies the data in [`experiments/language/dataset_store/LICENSE-MedMCQA.md`](experiments/language/dataset_store/LICENSE-MedMCQA.md).
- **Cite:** Pal, A., Umapathi, L. K. & Sankarasubbu, M. (2022). MedMCQA: A large-scale multi-subject multi-choice dataset for medical domain question answering. *Proceedings of the Conference on Health, Inference, and Learning*, PMLR 174, 248–260.

### Brain Tumor MRI (images in figures and notebook outputs)

- **Where:** rendered MRI slices appear in `experiments/results/lime_visual_examples/*.pdf` and in the saved outputs of `experiments/mri_lime_visual_demo.ipynb`. The dataset itself is not included.
- **Source:** Nickparvar, M. (2021). *Brain Tumor MRI Dataset*. Kaggle. <https://www.kaggle.com/dsv/2645886>
- **Licence:** **to be confirmed before public release.** The dataset is assembled from earlier collections (figshare, SARTAJ and Br35H), whose terms may also apply.

## Download separately

### PhysioNet/CinC Challenge 2012

- **Source:** <https://physionet.org/content/challenge-2012/1.0.0/> (training set A and `Outcomes-a.txt`)
- **Licence:** [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/), open access.
- **Preparation:** `experiments/tabular/process_physionet_data.py` builds the one-row-per-patient table `PhysionetChallenge2012-set-a.csv.gz` from the raw set A files and splits it into the missingness-level files the benchmark reads from `$MCAL_DATA_ROOT/tabular/missingness_levels/`.
- **Cite:** Silva, I., Moody, G., Scott, D. J., Celi, L. A. & Mark, R. G. (2012). Predicting in-hospital mortality of ICU patients: The PhysioNet/Computing in Cardiology Challenge 2012. *Computing in Cardiology*, 39, 245–248. PhysioNet also asks you to cite the platform, as listed on the page above.

### Breast Cancer Wisconsin (Diagnostic)

- **Source:** loaded at run time through scikit-learn's `load_breast_cancer`; nothing is redistributed. Original: UCI Machine Learning Repository, <https://doi.org/10.24432/C5DW2B>
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Cite:** Wolberg, W., Mangasarian, O., Street, N. & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

### CheXpert

- **Source:** Stanford AIMI, <https://stanfordmlgroup.github.io/competitions/chexpert/>. The experiments use the downsampled `CheXpert-v1.0-small` release; see `experiments/vision/chexpert_data_setup.py` for where it is expected.
- **Licence:** Stanford's CheXpert research use agreement, which does not permit redistribution. Never commit CheXpert images or labels; the directory is excluded in `.gitignore`.
- **Cite:** Irvin, J., Rajpurkar, P., Ko, M., et al. (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 590–597.

### BreakHis

- **Source:** <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>
- **Licence:** research use, under the providers' terms.
- **Cite:** Spanhol, F. A., Oliveira, L. S., Petitjean, C. & Heutte, L. (2016). A dataset for breast cancer histopathological image classification. *IEEE Transactions on Biomedical Engineering*, 63(7), 1455–1462.

### MedQA

- **Source:** <https://github.com/jind11/MedQA>
- **Licence:** MIT
- **Cite:** Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H. & Szolovits, P. (2021). What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Applied Sciences*, 11(14), 6421.

## Models

- **Vision:** ViT-B/16 from [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0). The patch-dropping variant in `vit_patch_drop/` is third-party code; see [`vit_patch_drop/NOTICE`](vit_patch_drop/NOTICE). Trained weights are not included.
- **Language:** Llama weights are not included and are subject to Meta's Llama licence.
