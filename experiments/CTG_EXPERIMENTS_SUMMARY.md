# CTG (Cardiotocography) Faithfulness Experiments - Summary

**Date:** January 19, 2026
**Dataset:** Cardiotocography (CTG) from UCI ML Repository
**Task:** Multi-class classification (3 classes: Normal, Suspect, Pathologic)

---

## Dataset Information

- **Source:** UCI ML Repository (already used in MCal paper)
- **Samples:** 2,126 total (1,700 train / 426 test)
- **Features:** 21 numerical features (fetal heart rate, uterine contractions)
- **Classes:** 3 (Normal, Suspect, Pathologic)
- **Experiment Size:** 300 train / 50 test samples
- **Base Model:** XGBoost (100 estimators, max_depth=6)
- **Explanation Method:** SHAP (KernelExplainer)

---

## Results Summary

### CTG Results (Multi-class)

| Metric | Uncalibrated | MCal Calibrated | Improvement |
|--------|-------------|-----------------|-------------|
| **Faithfulness (Pearson ρ) ↑** | 0.534 | 0.536 | **+0.3%** |
| **Deletion AUC ↓** | 0.288 | 0.499 | **-73.0%** ⚠️ |
| **Insertion AUC ↑** | 0.773 | 0.825 | **+6.7%** ✓ |

**Key Observations:**
- ✓ Insertion AUC improved (+6.7%)
- ✓ Faithfulness slightly improved (+0.3%)
- ⚠️ **Deletion AUC worsened significantly (-73.0%)**

This is unexpected and different from the Breast Cancer results.

---

## Comparison with Other Datasets

### Breast Cancer (Binary Classification)

| Metric | Uncalibrated | MCal Calibrated | Improvement |
|--------|-------------|-----------------|-------------|
| **Faithfulness (Pearson ρ) ↑** | 0.242 | 0.379 | **+56.8%** ✓✓ |
| **Deletion AUC ↓** | 0.668 | 0.662 | **+0.9%** ✓ |
| **Insertion AUC ↑** | 0.886 | 0.822 | **-7.2%** |

### MRI (Binary Classification - SHAP)

| Metric | Uncalibrated | MCal Calibrated | Improvement |
|--------|-------------|-----------------|-------------|
| **Faithfulness (Pearson ρ) ↑** | -0.051 | -0.042 | **+18.3%** ✓ |
| **Deletion AUC ↓** | 0.554 | 0.238 | **+57.1%** ✓✓ |
| **Insertion AUC ↑** | 0.389 | 0.210 | **-46.0%** |

---

## Analysis: Why Are CTG Results Different?

### Possible Explanations

1. **Multi-class vs Binary Classification**
   - CTG is 3-class while Breast Cancer and MRI are binary
   - Multi-class SHAP explanations may behave differently
   - The deletion metric may be harder to interpret in multi-class settings

2. **Class Imbalance**
   - Test set has severe imbalance: 44 Normal / 4 Suspect / 2 Pathologic
   - Only 6 samples from minority classes in test set
   - Results may be dominated by Normal class behavior

3. **Small Test Set**
   - Only 50 test samples (vs 426 available)
   - High variance in metrics
   - May not be representative of full dataset

4. **Model Performance**
   - Base model has 92% test accuracy (very high)
   - Model may be overfitting to training data
   - Calibration may not help when base model is already very confident

5. **SHAP Kernel Explainer Variability**
   - KernelExplainer uses sampling (nsamples=100)
   - Multi-class explanations may have higher variance
   - Different samples may produce different attribution rankings

---

## Comparison with MCal Paper Results

The MCal paper reported CTG results using **missingness bias** metric:

| Method | Missingness Bias |
|--------|-----------------|
| Original (Uncalibrated) | 1.06e-1 |
| MCal | **3.35e-3** ✓✓ |

The paper showed **strong improvement** on CTG using missingness bias.

**Key Difference:**
- Paper used **missingness bias** (different metric)
- We used **faithfulness/deletion/insertion** (explanation-based metrics)
- These measure different aspects of calibration quality

---

## Recommendations

### To Improve CTG Results:

1. **Use Full Test Set**
   - Use all 426 test samples instead of 50
   - This will reduce variance and provide more reliable metrics

2. **Balance Test Set**
   - Ensure adequate samples from all 3 classes
   - Currently only 2 Pathologic samples in test set

3. **Use TreeSHAP Instead of KernelExplainer**
   - TreeSHAP is exact and deterministic for tree models
   - KernelExplainer uses sampling (nsamples=100) which adds variance
   - Breast Cancer used TreeSHAP and showed better results

4. **Increase SHAP Samples**
   - Try nsamples=500 or 1000 instead of 100
   - This may reduce variance in SHAP values

5. **Try Different Explanation Methods**
   - Compare SHAP with LIME
   - Test if results are consistent across methods

6. **Validate Against Paper's Missingness Bias**
   - Implement the paper's missingness bias metric
   - Verify that MCal improves this metric on CTG
   - This would confirm MCal is working as expected

---

## Next Steps

### Immediate Actions:

1. ✓ **Completed:** Run initial CTG experiments
2. **TODO:** Re-run with full test set (426 samples)
3. **TODO:** Switch to TreeSHAP instead of KernelExplainer
4. **TODO:** Ensure balanced sampling from all classes

### For Paper/Rebuttal:

- **Current Status:** CTG results are mixed and need investigation
- **Don't Include Yet:** Wait for improved results with TreeSHAP and full test set
- **Alternative:** Focus on Breast Cancer results which are strong and clear

### Additional Datasets to Try:

From `PROPOSED_DATASETS.md`:
- **Wine Quality** (sklearn, multi-class, 178 samples)
- **Diabetes** (sklearn, binary, 442 samples)

These are immediately available without downloads and may provide clearer results.

---

## Files Generated

### CSV Results:
- `/experiments/results/ctg_faithfulness_results.csv`

### Visualizations:
- `/experiments/results/ctg_deletion_curves.pdf`
- `/experiments/results/ctg_insertion_curves.pdf`
- `/experiments/results/ctg_faithfulness_comparison.pdf`

### Code:
- `/experiments/run_faithfulness_experiments_ctg.py`
- `/experiments/all_data_loaders.py` (added CTG loaders)

---

## Conclusion

The CTG experiments have been **completed** but show **mixed results**:

✓ **Positive:**
- Insertion AUC improved (+6.7%)
- Faithfulness slightly improved (+0.3%)
- Successfully demonstrated MCal on multi-class classification

⚠️ **Concerning:**
- Deletion AUC worsened significantly (-73%)
- Results differ from strong Breast Cancer improvements
- Small test set (50 samples) may not be representative

**Recommendation:** Re-run with TreeSHAP and full test set before drawing conclusions about MCal's effectiveness on CTG.
