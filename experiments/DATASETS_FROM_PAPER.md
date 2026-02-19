# Datasets Used in MCal Paper

Based on Section 4 of the MCal paper, here are all the datasets evaluated:

## Vision (3 datasets)
1. ✅ **Brain MRI** - Already done in our experiments
2. **CheXpert** (Chest X-ray) - Not done yet
3. **BreakHis** (Breast Cancer Histopathology) - Not done yet

## Language (2 datasets)
1. **MedQA** - Done in paper, not in our experiments
2. **MedMCQA** - Done in paper, not in our experiments

## Tabular (3 datasets)
1. ✅ **Breast Cancer (Wisconsin)** - Already done in our experiments
2. **PhysioNet** - Done in paper, not in our experiments
3. **Cardiotocography (CTG)** - Not done yet

---

## For Faithfulness Experiments - Priority Order

### Tabular Datasets to Run:

#### 1. Cardiotocography (CTG) ⭐ **START HERE**
- **Source:** UCI ML Repository (already referenced in paper)
- **Task:** Multi-class classification (fetal state)
- **Features:** 21 numerical features (fetal heart rate, uterine contractions)
- **Samples:** ~2,126 samples
- **Classes:** 3 or 10 (depends on classification scheme)
- **Domain:** Medical (obstetrics/fetal monitoring)
- **Reference in paper:** Campos & Bernardes, 2000
- **UCI Link:** Already cited: https://doi.org/10.24432/C51S4N
- **Why:** Medical domain, already validated in MCal paper

#### 2. PhysioNet (Optional - if needed)
- **Source:** PhysioNet Challenge
- **Task:** Binary classification (mortality prediction)
- **Domain:** Medical (ICU data)
- **Note:** Paper already has results for this - we could replicate

---

## Implementation Plan for CTG

### Step 1: Download and Load CTG Dataset
The CTG dataset should be available through:
- UCI ML Repository
- Or potentially sklearn/ucimlrepo Python package

### Step 2: Preprocessing
- Load 21 numerical features
- Handle class labels (use 3-class or 10-class version)
- Split train/test stratified
- Standardize features

### Step 3: Run Same Pipeline as Breast Cancer
- Train XGBoost classifier
- Train MCal calibrator on ablated data
- Generate TreeSHAP explanations
- Compute Faithfulness, Deletion, Insertion metrics
- Create visualizations
- Save results

### Step 4: Compare with Paper Results
The paper reports these CTG results (Table 1):
- Original (uncalibrated): 1.06e-1
- MCal: 3.35e-3 (missingness bias metric)

We should get similar results and add our faithfulness metrics.

---

## Why Start with CTG?

1. ✅ **Already validated in MCal paper** - We know MCal works on this dataset
2. ✅ **Pure tabular** - Same modality as Breast Cancer
3. ✅ **Medical domain** - Aligns with paper focus
4. ✅ **Available in UCI** - Easy to download
5. ✅ **Numerical features** - TreeSHAP will work well
6. ✅ **Multi-class** - Different from binary Breast Cancer

---

## Expected Results for CTG

Based on Breast Cancer results (+56.8% faithfulness), we expect:
- **Faithfulness improvement:** +40-60%
- **Deletion metric:** Improvement (sign of better explanations)
- **Insertion metric:** Likely neutral or small improvement

This will give us:
- **Total tabular datasets:** 2 (Breast Cancer + CTG)
- **Coverage:** Binary + Multi-class classification
- **All from MCal paper** - Strong validation

---

## Next Steps

1. **Download CTG from UCI** or use Python package
2. **Create loader** in `all_data_loaders.py`
3. **Run experiments** using existing `run_faithfulness_experiments_tabular.py`
4. **Compare results** with paper and Breast Cancer

Should I proceed with CTG?
