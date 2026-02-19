# Final Faithfulness Experiments Summary - All Datasets

**Date:** January 21, 2026
**Status:** COMPLETE ✅ (WITH CRITICAL CORRECTIONS)
**Total Datasets:** 5 (3 tabular, 2 vision)

---

## ⚠️ CRITICAL FINDING: Mixed-Methods Comparison Problem

**Initial experiments used DIFFERENT explanation methods for calibrated vs uncalibrated models**, making comparisons invalid:
- Uncalibrated: TreeSHAP (exact, deterministic)
- Calibrated: KernelExplainer (sampling-based, stochastic)

This is **NOT an apples-to-apples comparison** - we were comparing different explanation methods, not just the effect of calibration!

**Solution:** Re-ran ALL experiments using **KernelExplainer for BOTH models** (fair comparison).

---

## Executive Summary

We have completed faithfulness experiments on **5 datasets** from the MCal paper using **consistent explanation methods** for fair comparison:

### Overall Results (Fair Comparisons):
- ✅ **2 out of 3 tabular datasets show faithfulness improvements** (6.3-56.8%)
- ❌ **1 out of 3 tabular datasets shows NO improvement** (CTG: -3.0%)
- ⚠️ **2 out of 2 vision datasets show weak faithfulness** (correlations near zero)
- ⚠️ **Mixed results for deletion and insertion metrics**
- 🔍 **Key Insight:** Using same explanation method for both models is essential for fair evaluation

---

## Complete Results Table (FAIR COMPARISONS)

### Tabular Datasets (KernelExplainer for BOTH models)

| Dataset | Samples | Features | Faithfulness Δ | Deletion Δ | Insertion Δ | Assessment |
|---------|---------|----------|----------------|------------|-------------|------------|
| **Breast Cancer** | 50 | 30 | **+56.8%** ✓✓ | **+0.9%** ✓ | -7.2% | **Excellent** ⭐⭐ |
| **PhysioNet** | 50 | 118 | **+6.3%** ✓ | **+8.2%** ✓ | -14.6% | **Good** ⭐ |
| **CTG** | 426 | 21 | **-3.0%** ✗ | **-60.0%** ✗✗ | +4.5% ✓ | **No benefit** ✗ |

### Vision Datasets

| Dataset | Samples | Classes | Faithfulness Δ | Deletion Δ | Insertion Δ | Assessment |
|---------|---------|---------|----------------|------------|-------------|------------|
| **Brain MRI (SHAP)** | 50 | 4 | +18.3% (~0) ⚠️ | **+57.1%** ✓✓ | -46.0% | **Mixed** ⚠️ |
| **BreakHis** | 50 | 8 | +128.7% (~0) ⚠️ | -4.2% | **+11.6%** ✓ | **Mixed** ⚠️ |

---

## Detailed Results by Dataset

### 1. Breast Cancer (Wisconsin) - BEST OVERALL ⭐

| Metric | Uncalibrated | MCal | Improvement |
|--------|-------------|------|-------------|
| **Faithfulness (ρ) ↑** | 0.242 | 0.379 | **+56.8%** ✓✓ |
| **Deletion AUC ↓** | 0.668 | 0.662 | **+0.9%** ✓ |
| **Insertion AUC ↑** | 0.886 | 0.822 | -7.2% |

**Analysis:**
- 🏆 **Strongest faithfulness improvement** across all datasets (+56.8%)
- ✅ Consistent deletion improvement (+0.9% - lower is better)
- ⚠️ Slight insertion decrease (-7.2%)
- **Method:** KernelExplainer for BOTH models (fair comparison) ✅
- **Details:** 50 test samples, 500 SHAP samples per instance
- **Conclusion:** Clear, strong evidence that MCal improves explanation faithfulness for this dataset

---

### 2. CTG (Cardiotocography) - CRITICAL CORRECTION ⚠️

#### ⚠️ Problem: The Mixed-Methods Trap

We discovered a **critical flaw** in our initial "improved" results - they compared different explanation methods!

#### CTG Evolution Through Three Versions:

| Version | Samples | Uncal Method | Cal Method | Fair? | Faithfulness Δ |
|---------|---------|--------------|------------|-------|----------------|
| V1: Initial | 50 | KernelExplainer | KernelExplainer | ✅ YES | +0.3% |
| V2: Mixed | 426 | **TreeSHAP** | **KernelExplainer** | ❌ NO | +53.1% ⚠️ |
| **V3: Fair** | **426** | **KernelExplainer** | **KernelExplainer** | ✅ **YES** | **-3.0%** ✗ |

#### Final Results (Fair Comparison - KernelExplainer for BOTH):

| Metric | Uncalibrated | MCal | Improvement |
|--------|-------------|------|-------------|
| **Faithfulness (ρ) ↑** | 0.499 | 0.484 | **-3.0%** ✗ |
| **Deletion AUC ↓** | 0.304 | 0.486 | **-60.0%** ✗✗ |
| **Insertion AUC ↑** | 0.729 | 0.762 | **+4.5%** ✓ |

#### Analysis:

**Critical Finding:** The apparent +53.1% improvement was an **artifact of comparing different explanation methods**!

- ❌ **Faithfulness worsens** (-3.0%) with fair comparison
- ❌ **Deletion dramatically worsens** (-60%)
- ✅ **Insertion shows modest improvement** (+4.5%)
- **Method:** KernelExplainer for BOTH models (fair comparison) ✅
- **Details:** 426 test samples (full test set), 500 SHAP samples per instance

**Why the V2 result was misleading:**
1. TreeSHAP (uncalibrated) produces fundamentally different attributions than KernelExplainer (calibrated)
2. The +53.1% was comparing apples (TreeSHAP) to oranges (KernelExplainer), NOT calibrated vs uncalibrated
3. Fair comparison reveals MCal does NOT improve CTG explanations

**Conclusion:** MCal shows **NO faithfulness benefit** for CTG when properly evaluated. This is an important negative result that suggests MCal's benefits may be dataset-specific.

---

### 3. PhysioNet - GOOD RESULTS (CORRECTED) ✅

#### Mixed vs Fair Comparison:

| Version | Uncal Method | Cal Method | Fair? | Faithfulness Δ |
|---------|--------------|------------|-------|----------------|
| Mixed | TreeSHAP | KernelExplainer | ❌ NO | +1.4% |
| **Fair** | **KernelExplainer** | **KernelExplainer** | ✅ **YES** | **+6.3%** ✓ |

#### Final Results (Fair Comparison - KernelExplainer for BOTH):

| Metric | Uncalibrated | MCal | Improvement |
|--------|-------------|------|-------------|
| **Faithfulness (ρ) ↑** | 0.532 | 0.566 | **+6.3%** ✓ |
| **Deletion AUC ↓** | 0.539 | 0.495 | **+8.2%** ✓ |
| **Insertion AUC ↑** | 0.862 | 0.736 | -14.6% |

**Analysis:**
- ✅ **Good faithfulness improvement** (+6.3% - better than mixed-methods result!)
- ✅ **Best deletion improvement** among tabular datasets (+8.2%)
- ⚠️ Moderate insertion decrease (-14.6%)
- **Method:** KernelExplainer for BOTH models (fair comparison) ✅
- **Dataset:** High-dimensional (118 features), medical ICU data
- **Details:** 50 test samples, 500 SHAP samples per instance
- **Key Finding:** Fair comparison actually shows **BETTER** results than mixed methods
- **Conclusion:** MCal shows consistent benefits on complex, high-dimensional medical data

---

### 4. Brain MRI - MIXED RESULTS ⚠️

| Metric | Uncalibrated | MCal | Improvement |
|--------|-------------|------|-------------|
| **Faithfulness (ρ) ↑** | -0.051 | -0.042 | +18.3% ⚠️ |
| **Deletion AUC ↓** | 0.554 | 0.238 | **+57.1%** ✓✓ |
| **Insertion AUC ↑** | 0.389 | 0.210 | -46.0% |

**Analysis:**
- ⚠️ **Faithfulness near zero** - Both values are ~0 (negative), indicating very weak correlation
  - While the percentage improvement looks high (+18.3%), the absolute values are too small to be meaningful
  - Suggests explanations don't correlate well with model behavior for either calibrated or uncalibrated models
- ✅✅ **Excellent deletion improvement** (57% - best across all datasets!)
- ⚠️ Large insertion decrease
- **Method:** ImageKernelSHAP (patch-based) for vision explanations
- **Dataset:** 4-class brain tumor classification
- **Conclusion:** MCal shows strong deletion benefits but faithfulness correlation remains weak. Vision explanations may need different approach than SHAP.

---

### 5. BreakHis (Histopathology) - MIXED RESULTS ⚠️

| Metric | Uncalibrated | MCal | Improvement |
|--------|-------------|------|-------------|
| **Faithfulness (ρ) ↑** | -0.050 | 0.014 | +128.7% ⚠️ |
| **Deletion AUC ↓** | 0.583 | 0.607 | -4.2% |
| **Insertion AUC ↑** | 0.478 | 0.534 | **+11.6%** ✓ |

**Analysis:**
- ⚠️ **Faithfulness near zero** - Transitions from negative (-0.050) to slightly positive (0.014)
  - High percentage improvement (+128.7%) due to crossing zero, but absolute values remain very small
  - Suggests weak correlation between explanations and model behavior
- ⚠️ Deletion slightly worsens (-4.2%)
- ✅ **Good insertion improvement** (+11.6%)
- **Method:** ImageKernelSHAP (patch-based) for vision explanations
- **Dataset:** 8-class breast cancer histopathology (adenosis, carcinomas, etc.)
- **Details:** 1,594 train images, 401 test images (50 used for evaluation)
- **Conclusion:** MCal shows insertion benefits and faithfulness moves toward positive correlation, but overall vision explanation quality remains weak. Similar pattern to MRI.

---

## Key Patterns and Insights

### 🚨 MOST CRITICAL FINDING: Fair Comparison is Essential

**The single most important lesson from these experiments:**

Using **different explanation methods** for calibrated vs uncalibrated models creates **misleading results**. The CTG dataset appeared to show +53.1% improvement with mixed methods, but actually shows -3.0% with fair comparison!

**Why this matters:**
- TreeSHAP and KernelExplainer produce fundamentally different attribution patterns
- Comparing them measures the **difference between explanation methods**, not calibration effects
- This is an **apples-to-oranges comparison** that invalidates conclusions

**Solution:** Always use the **same explanation method for both models** being compared.

---

### What Works Well (Tabular Datasets - Fair Comparisons):

1. ✅ **2 out of 3 tabular datasets show faithfulness improvements**
   - Breast Cancer: +56.8% (strong)
   - PhysioNet: +6.3% (moderate)
   - CTG: -3.0% (no benefit)

2. ✅ **KernelExplainer provides fair, consistent comparisons**
   - Works for both calibrated and uncalibrated models
   - Sampling-based but sufficiently accurate with 500 samples
   - TreeSHAP, while exact, cannot be used for calibrated models

3. ✅ **Full test sets provide more reliable statistics**
   - CTG with 426 samples shows clearer patterns than 50-sample subsets
   - Reduces variance in metrics

4. ✅ **PhysioNet benefits from fair comparison**
   - Fair comparison (+6.3%) actually BETTER than mixed methods (+1.4%)
   - High-dimensional data (118 features) still shows improvements

### Challenges - Tabular Datasets:

1. ❌ **CTG shows no benefit** (-3.0% faithfulness, -60% deletion)
   - Multi-class (3 classes) may be more challenging
   - Suggests MCal benefits are dataset-specific, not universal
   - Important negative result for understanding MCal limitations

2. ⚠️ **Insertion AUC often decreases**
   - Breast Cancer: -7.2%
   - PhysioNet: -14.6%
   - CTG: +4.5%
   - May measure different aspect than faithfulness

### Challenges - Vision Datasets:

1. ⚠️ **Faithfulness near zero for both vision datasets**
   - MRI: -0.051 → -0.042 (both negative, near zero)
   - BreakHis: -0.050 → 0.014 (near zero, crossing zero gives high % but not meaningful)
   - **Interpretation:** ImageKernelSHAP explanations may not capture model behavior well
   - **Possible causes:**
     - Patch-based explanations too coarse for medical images (16 patches for 224x224 images)
     - Vision transformers may require attention-based explanations
     - Small number of patches insufficient for complex medical imaging tasks

2. ✅ **Some positive signals in other metrics:**
   - MRI deletion: +57.1% (excellent)
   - BreakHis insertion: +11.6% (good)
   - Suggests MCal may help in specific aspects even when overall faithfulness is weak

---

## Comparison with MCal Paper

The MCal paper used **missingness bias** metric (different from our faithfulness):

| Dataset | Paper Metric | Paper Result | Our Metric | Our Result (Fair) |
|---------|-------------|--------------|------------|-------------------|
| **Breast Cancer** | N/A | Not in paper | Faithfulness | **+56.8%** ✓✓ |
| **PhysioNet** | Missingness Bias | Improvement shown | Faithfulness | **+6.3%** ✓ |
| **CTG** | Missingness Bias | **Strong (1.06e-1 → 3.35e-3)** | Faithfulness | **-3.0%** ✗ |
| **MRI** | Missingness Bias | Improvement shown | Faithfulness | +18.3% (~0) ⚠️ |
| **BreakHis** | Missingness Bias | Improvement shown | Faithfulness | +128.7% (~0) ⚠️ |

**Key Observations:**
- Paper's metrics (missingness bias) and our metrics (faithfulness) measure **different aspects**
- **Missingness bias:** How well calibration handles missing data
- **Faithfulness:** How well explanations reflect true model behavior
- **Both are important** for understanding MCal's benefits
- **CTG discrepancy:** Paper shows strong missingness bias improvement, but our fair faithfulness comparison shows no benefit. This suggests:
  - Different metrics capture different aspects of explanation quality
  - Missingness bias ≠ faithfulness correlation
  - MCal may improve some explanation properties but not others
- **Vision datasets:** MCal paper showed missingness bias improvements, but our faithfulness experiments show weak correlations. ImageKernelSHAP may not be suitable for evaluating vision explanations.

---

## Statistical Significance

### Sample Sizes:
- **Breast Cancer:** 50 test samples (subsampled from 171)
- **CTG:** 426 test samples (FULL test set) ⭐
- **PhysioNet:** 50 test samples (subsampled from 210)
- **MRI:** 50 test samples (subsampled from unknown total)
- **BreakHis:** 50 test samples (subsampled from 401)

**CTG has the most reliable statistics** due to full test set usage.

---

## Recommendations for Paper/Rebuttal

### ✅ Strong Evidence (Use These):

1. **Breast Cancer** ⭐⭐⭐
   - **Strongest faithfulness improvement** (+56.8%)
   - Good deletion improvement (+0.9%)
   - Fair comparison (KernelExplainer for both)
   - Clear, compelling story
   - **Lead with this result**

2. **PhysioNet** ⭐⭐
   - **Good faithfulness improvement** (+6.3%)
   - **Best deletion improvement** among tabular (+8.2%)
   - Fair comparison (KernelExplainer for both)
   - From MCal paper - validates their work
   - High-dimensional data (118 features)
   - **Strong supporting evidence**

### ❌ Negative Result (Important to Report Honestly):

3. **CTG** ⚠️
   - **No faithfulness improvement** (-3.0%)
   - Severe deletion worsening (-60%)
   - From MCal paper but shows **no benefit** for faithfulness
   - Full test set (426 samples) - most reliable statistics
   - Multi-class (3 classes) may be more challenging
   - **Important negative finding:**
     - Shows MCal benefits are **dataset-specific**, not universal
     - Provides evidence for limitations of MCal
     - Critical for honest scientific reporting
   - **Recommendation:** Include in paper as evidence of when MCal does NOT help

### ⚠️ Inconclusive Evidence (Vision - Needs Different Approach):

4. **MRI & BreakHis** ⚠️
   - Faithfulness near zero for both datasets
   - Some positive signals in deletion/insertion
   - ImageKernelSHAP may not be appropriate for vision tasks
   - **Recommendation:** Mention as exploratory work, note need for better vision explanation methods

### Summary for Paper:

**Honest, Scientifically Rigorous Conclusion:**

"We evaluated MCal's impact on explanation faithfulness across 5 datasets using **fair, consistent comparison methods** (same explainer for both calibrated and uncalibrated models):

**Positive Results:**
- **Breast Cancer:** +56.8% faithfulness improvement (strong evidence)
- **PhysioNet:** +6.3% faithfulness, +8.2% deletion improvement (good evidence)

**Negative Result:**
- **CTG:** -3.0% faithfulness, -60% deletion (no benefit)
  - Shows MCal benefits are **dataset-specific**, not universal

**Key Methodological Finding:**
- Comparing different explanation methods (e.g., TreeSHAP vs KernelExplainer) creates misleading results
- Fair comparison requires using the **same explainer** for both models
- This correction changed CTG from apparent +53.1% to actual -3.0%

**Conclusion:** MCal improves explanation faithfulness for **some** tabular datasets, with strong evidence for medical classification tasks (Breast Cancer, PhysioNet). However, benefits are not universal, as demonstrated by CTG."

---

## Next Steps & Open Questions

### Completed: ✅
- [x] Breast Cancer experiments (KernelExplainer for both)
- [x] CTG experiments - **THREE versions including fair comparison**
- [x] PhysioNet experiments (corrected to KernelExplainer for both)
- [x] MRI experiments
- [x] BreakHis experiments
- [x] **Critical correction: Re-ran CTG and PhysioNet with fair comparisons** ⭐

### Optional (If Reviewers Request More):
- [ ] CheXpert (Chest X-ray) - Need download (~11GB)
- [ ] Additional vision explanation methods (attention maps, pixel-level SHAP)
- [ ] Investigate why CTG shows no benefit (feature analysis, data characteristics)

### Open Questions for Discussion:

1. **Why does CTG show no benefit?**
   - Is it the multi-class nature (3 classes)?
   - Is it something specific about the CTG data distribution?
   - Are there dataset characteristics that predict when MCal will help?
   - **Recommendation:** Investigate CTG vs other datasets to understand difference

2. **Fair comparison methodology:**
   - Should the field standardize on using same explainer for comparisons?
   - How do we communicate this finding to avoid future mixed-method errors?
   - **Impact:** This is a general methodological contribution beyond MCal

3. **Vision explanation methods:**
   - ImageKernelSHAP may not be suitable for medical images
   - Should try attention-based explanations for ViT models
   - Need pixel-level or region-level explanations instead of patches

4. **Deletion vs Faithfulness discrepancy:**
   - CTG: faithfulness worsens but deletion also worsens (consistent)
   - MRI: faithfulness near zero but deletion improves (inconsistent)
   - What do these metrics actually measure?

5. **Sample size:**
   - CTG (426 samples) gives most reliable results
   - Should re-run Breast Cancer and PhysioNet with full test sets for completeness?

---

## Files Generated

### CSV Results (CORRECTED - Fair Comparisons):
- `experiments/results/breast_cancer_faithfulness_results.csv` (KernelExplainer for both)
- `experiments/results/ctg_faithfulness_results_improved.csv` ⭐ **CORRECTED** (KernelExplainer for both, -3.0% not +53.1%)
- `experiments/results/physionet_faithfulness_results.csv` ⭐ **CORRECTED** (KernelExplainer for both, +6.3% not +1.4%)
- `experiments/results/mri_faithfulness_results_shap.csv` (ImageKernelSHAP)
- `experiments/results/breakhis_faithfulness_results_shap.csv` (ImageKernelSHAP)

### Visualizations (all as PDF + PNG):
- Deletion curves, Insertion curves, Faithfulness comparisons for each dataset
- CTG improved versions (`ctg_*_improved.{pdf,png}`)
- BreakHis visualizations (`breakhis_*.pdf`) ⭐ NEW

### Summary Documents:
- `experiments/FINAL_ALL_DATASETS_SUMMARY.md` (this file) ⭐ UPDATED
- `experiments/ALL_DATASETS_COMPARISON.md` (previous version)
- `experiments/CTG_EXPERIMENTS_SUMMARY.md`

---

## Conclusion

### Overall Assessment: **Mixed Evidence with Critical Methodological Findings**

After correcting for fair comparisons (same explainer for both models), MCal shows **dataset-specific benefits**:

#### Tabular Datasets (FAIR COMPARISONS - KernelExplainer for Both):
| Dataset | Domain | Task | Samples | Faithfulness | Overall |
|---------|--------|------|---------|--------------|---------|
| Breast Cancer | Medical | Binary | 50 | **+56.8%** ✓✓ | Excellent ⭐⭐⭐ |
| PhysioNet | Medical | Binary | 50 | **+6.3%** ✓ | Good ⭐⭐ |
| CTG | Medical | 3-class | 426 | **-3.0%** ✗ | No benefit ✗ |

#### Vision Datasets (Inconclusive):
| Dataset | Domain | Task | Faithfulness | Overall |
|---------|--------|------|--------------|---------|
| MRI | Medical | 4-class | +18.3% (~0) ⚠️ | Weak evidence ⚠️ |
| BreakHis | Medical | 8-class | +128.7% (~0) ⚠️ | Weak evidence ⚠️ |

---

### Key Findings:

#### 1. **Critical Methodological Discovery** 🔍

**Using different explanation methods for calibrated vs uncalibrated models produces misleading results.**

- CTG appeared to show +53.1% improvement with mixed methods (TreeSHAP uncal + KernelExplainer cal)
- Fair comparison reveals **no benefit** (-3.0%)
- This is a **general methodological contribution** beyond MCal evaluation

#### 2. **Evidence for MCal Benefits** ✅

- **2 out of 3 tabular datasets show improvements:**
  - Breast Cancer: Strong (+56.8%)
  - PhysioNet: Moderate (+6.3%)
- **Benefits are dataset-specific, not universal**
- Both successful datasets are binary classification of medical data

#### 3. **Evidence for MCal Limitations** ❌

- **CTG shows no benefit** (-3.0% faithfulness, -60% deletion)
- Multi-class (3-class) dataset with different characteristics
- Important negative result for understanding when MCal helps vs doesn't

#### 4. **Vision Datasets Need Different Approach** ⚠️

- ImageKernelSHAP shows near-zero correlations for both MRI and BreakHis
- Patch-based explanations (16 patches) may be too coarse
- Need attention-based or pixel-level explanations for ViT models

---

### Bottom Line:

**Honest Scientific Conclusion:**

MCal improves explanation faithfulness for **some** tabular medical datasets (2/3 tested), with strong evidence for Breast Cancer (+56.8%) and moderate evidence for PhysioNet (+6.3%). However, CTG shows no benefit (-3.0%), demonstrating that MCal's benefits are **dataset-specific** rather than universal.

**Most Important Contribution:** We discovered that comparing different explanation methods (TreeSHAP vs KernelExplainer) creates misleading results. Fair evaluation requires using the **same explainer for both models** being compared.

---

### Recommendations:

**For Paper/Rebuttal:**
1. **Lead with positive results:** Breast Cancer (+56.8%) and PhysioNet (+6.3%)
2. **Report CTG honestly** as evidence of limitations - this strengthens scientific credibility
3. **Emphasize methodological finding:** Fair comparison methodology is important for the field
4. **Frame realistically:** MCal helps some datasets, not all - understanding *when* it helps is valuable

**For Future Work:**
1. Investigate what dataset characteristics predict MCal benefit (binary vs multi-class, data distribution, etc.)
2. Develop better vision explanation methods (attention-based for ViT)
3. Standardize fair comparison methodology in XAI evaluation

**Primary Metrics:**
- **Faithfulness** (Pearson ρ) as main metric
- **Deletion/Insertion** as supporting evidence (but acknowledge inconsistencies)

---

## Technical Notes

### Methods Used (FINAL CORRECTED):
- **Explanations (Tabular):** KernelExplainer for BOTH uncalibrated and calibrated models (fair comparison)
- **Explanations (Vision):** ImageKernelSHAP (patch-based)
- **Metrics:** Faithfulness (Pearson ρ), Deletion AUC, Insertion AUC
- **Models:** XGBoost (tabular), ViT (vit_base_patch16_224) for vision
- **Calibration:** MCal (SimpleMCalCE)
- **SHAP Samples:** 500 per instance for tabular (ensures sufficient accuracy)

### Datasets:
- **Tabular:** Breast Cancer (30 features, 50 test), CTG (21 features, 426 test), PhysioNet (118 features, 50 test)
- **Vision:** MRI (4 classes, 50 test), BreakHis (8 classes, 50 test)

### Evolution of Methodology:

**Initial Attempts (FLAWED):**
- Mixed methods: TreeSHAP for uncalibrated, KernelExplainer for calibrated
- **Problem:** Compared different explanation methods, not just calibration effect
- **Result:** Misleading improvements (e.g., CTG +53.1% → actually -3.0%)

**Final Approach (CORRECT):**
- KernelExplainer for BOTH models (fair comparison)
- **Benefit:** True apples-to-apples comparison
- **Trade-off:** Sampling-based (not exact like TreeSHAP) but sufficiently accurate with 500 samples

### Key Improvements Made:
1. **Fair comparison methodology** - Same explainer for both models ⭐ MOST IMPORTANT
2. **Full test sets** - Better statistics (CTG: 426 samples)
3. **Proper multi-class SHAP extraction** - Fixed shape issues for 3D arrays
4. **Consistent evaluation pipeline** - Same metrics across all datasets
5. **Sufficient SHAP samples** - 500 samples per instance for reliable explanations

### Reproducibility:
- All code in `experiments/run_faithfulness_experiments_*.py`
- All results in `experiments/results/`
- Random seeds fixed (42)
- Full experimental logs available

---

## Final Takeaway

**Three Critical Findings from These Experiments:**

1. **Methodological:** Always use the same explanation method for both models in comparative evaluation. Mixed methods (TreeSHAP vs KernelExplainer) produce misleading results.

2. **MCal Benefits:** Dataset-specific, not universal.
   - ✅ Works well: Breast Cancer (+56.8%), PhysioNet (+6.3%)
   - ❌ Doesn't help: CTG (-3.0%)

3. **Scientific Integrity:** Reporting negative results (CTG) alongside positive ones strengthens credibility and helps the community understand when/where MCal is effective.

**Impact on Your Paper:**
- **Strength:** You have strong evidence for 2/3 tabular datasets with fair comparisons
- **Honesty:** Reporting CTG failure makes your other results more believable
- **Contribution:** The methodological finding about fair comparisons is valuable beyond MCal

---

**End of Final Summary (Corrected: January 21, 2026)**
