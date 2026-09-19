# Patch-Based SHAP for Image Explanations

## Overview

This document explains the patch-based SHAP implementation used in MCal faithfulness experiments, and how it compares to LIME.

## Motivation

Both LIME and SHAP are popular explanation methods, but they operate differently:
- **LIME**: Local linear approximation using perturbed samples
- **SHAP**: Game-theoretic approach using Shapley values

By using the same patch-based representation for both methods, we can make a fair comparison of how MCal improves explanation quality across different explanation paradigms.

## Implementation

### Patch Representation
Both explainers use the same 16-patch representation:
- Image size: 224×224 pixels
- Patch size: 56×56 pixels
- Number of patches: 16 (4×4 grid)

This reduces the feature space from 50,176 pixels (224×224) to just 16 patches, making the problem tractable while preserving spatial structure.

### ImageLIME
```python
class ImageLIME(BaseExplainer):
    def explain_instance(self, image, label):
        1. Generate random perturbations (patch presence/absence)
        2. Get model predictions for perturbed images
        3. Fit weighted linear regression
        4. Return coefficients as patch importances
```

**Key characteristics:**
- Local approximation: explains one prediction at a time
- Perturbations: random sampling of patch combinations
- Weighting: based on number of patches kept (similarity to original)
- Output: Linear coefficients indicating patch importance

### ImageKernelSHAP (Patch-SHAP)
```python
class ImageKernelSHAP(BaseExplainer):
    def explain_instance(self, image, label):
        1. Generate coalitions (patch subsets)
        2. Evaluate model on each coalition
        3. Apply Shapley kernel weights
        4. Solve weighted least squares for SHAP values
        5. Return SHAP values as patch importances
```

**Key characteristics:**
- Game-theoretic: based on Shapley values from cooperative game theory
- Coalitions: systematic exploration of feature combinations
- Weighting: Shapley kernel weights (higher for empty/full coalitions)
- Output: SHAP values indicating patch contribution to prediction

## Theoretical Differences

### LIME Weighting
LIME uses distance-based weighting:
```
w_i = (# patches kept) / (total # patches)
```
This favors perturbations similar to the original image.

### SHAP Weighting (Shapley Kernel)
SHAP uses game-theoretic weights:
```
w(z) = (M-1) / (|z| * (M - |z|))
```
where:
- M = total number of features (16 patches)
- |z| = number of features present in coalition

This gives infinite weight to empty and full coalitions, ensuring desirable properties.

## Desirable Properties

### LIME
- ✓ Local fidelity (approximates model locally)
- ✓ Interpretability (sparse linear model)
- ✗ No guarantee of consistency
- ✗ Random sampling can miss important interactions

### SHAP
- ✓ Local accuracy (matches model at original point)
- ✓ Consistency (if feature contributes more, it gets higher value)
- ✓ Missingness (correctly handles feature absence)
- ✓ Theoretical foundation (unique solution satisfying axioms)
- ✗ More computationally intensive

## MCal and Explanation Quality

MCal addresses a key issue in both explainers: **missingness bias**.

When we mask patches to create perturbations:
- The model sees unusual inputs (masked patches = black regions)
- This creates distribution shift between training and explanation
- Explanations may not reflect true model behavior

**MCal's solution:**
1. Train on ablated (masked) data
2. Calibrate model to handle missing features correctly
3. Explanations now reflect actual model behavior under feature absence

This should improve faithfulness metrics for **both** LIME and SHAP.

## Experimental Design

### Metrics Computed
1. **Faithfulness (Pearson ρ)**: Correlation between attribution scores and actual prediction changes
2. **Deletion AUC**: Area under curve when removing patches by importance
3. **Insertion AUC**: Area under curve when adding patches by importance

### Expected Results
If MCal improves explanation quality:
- ✓ Faithfulness should increase (stronger correlation)
- ✓ Deletion AUC should decrease (removing important patches hurts more)
- ✓ Insertion AUC should increase (adding important patches helps more)

These improvements should be observed for **both** LIME and SHAP, demonstrating that MCal's benefits are explanation-method agnostic.

## Comparison to Previous Experiments

### Tabular (Breast Cancer)
- Used SHAP with TreeExplainer
- Features: 30 numerical features
- Already tractable without dimensionality reduction

### Vision (MRI) - Previous
- Used LIME with 16 patches
- Demonstrated MCal improvements on LIME explanations

### Vision (MRI) - Current
- Added SHAP with same 16 patches
- Now we can compare LIME vs SHAP
- Shows MCal works across different explanation paradigms

## Advantages of Patch-SHAP

1. **Consistency**: Same granularity as LIME (16 patches)
2. **Theoretical rigor**: Game-theoretic foundation
3. **Reproducibility**: SHAP values are deterministic (given same samples)
4. **Additivity**: SHAP values sum to prediction difference from baseline
5. **Completeness**: Satisfies desirable axioms (local accuracy, missingness, consistency)

## Computational Complexity

### Time per image:
- **LIME**: ~1-2 seconds (100 random perturbations)
- **SHAP**: ~10-11 seconds (100 coalitions + Shapley weights)

SHAP is slower but provides stronger theoretical guarantees.

### Total experiment time:
- 40 images × 2 models × (LIME + SHAP)
- LIME: ~160 seconds (~3 minutes)
- SHAP: ~880 seconds (~15 minutes)
- Plus metrics computation and visualization

## Code Location

- Implementation: `experiments/explanations.py` (`ImageLIME`, `ImageKernelSHAP`)

- Experiment: `python -m experiments.run_faithfulness --dataset mri`
  (code in `experiments/faithfulness/mri.py`)
  - Generates both LIME and SHAP explanations
  - Computes metrics for both
  - Creates separate visualizations

- Results are saved to:
  - `experiments/results/mri_faithfulness_results_lime.csv`
  - `experiments/results/mri_faithfulness_results_shap.csv`

## References

1. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP paper)
2. Ribeiro et al. (2016). "Why Should I Trust You?" (LIME paper)
3. Shapley, L. S. (1953). "A value for n-person games" (original Shapley values)

## Summary

Patch-SHAP provides a theoretically grounded alternative to LIME for image explanations. By using the same patch-based representation, we can fairly compare both methods and demonstrate that MCal improves explanation quality across different explanation paradigms. This strengthens the claim that MCal addresses a fundamental issue (missingness bias) that affects all perturbation-based explanation methods.
