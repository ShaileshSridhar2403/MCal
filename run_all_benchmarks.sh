#!/bin/bash

# MCal Benchmarks - Run all 8 datasets with 10 runs each
# This script runs all calibration benchmarks across vision, language, and tabular modalities

set -e  # Exit on error

echo "======================================================================"
echo "Starting MCal Benchmarks - All Datasets (10 runs each)"
echo "======================================================================"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Track start time
START_TIME=$(date +%s)

# ======================================================================
# VISION MODALITY (3 datasets)
# ======================================================================

echo "======================================================================"
echo "VISION MODALITY"
echo "======================================================================"
echo ""

# 1. Brain MRI
echo "----------------------------------------------------------------------"
echo "[1/8] Running Brain MRI benchmark..."
echo "----------------------------------------------------------------------"
cd "${SCRIPT_DIR}/experiments/vision"
python mri_kl_benchmark.py \
    --methods baseline replace_mean patchcutout arch_mod temperature platt mcal_ce \
    --runs 10
echo "✓ Brain MRI completed"
echo ""

# 2. CheXpert
echo "----------------------------------------------------------------------"
echo "[2/8] Running CheXpert benchmark..."
echo "----------------------------------------------------------------------"
cd "${SCRIPT_DIR}/experiments/vision"
python chexpert_kl_benchmark.py \
    --methods baseline replace_mean patchcutout arch_mod temperature platt mcal_ce \
    --runs 10
echo "✓ CheXpert completed"
echo ""

# 3. BreakHis
echo "----------------------------------------------------------------------"
echo "[3/8] Running BreakHis benchmark..."
echo "----------------------------------------------------------------------"
cd "${SCRIPT_DIR}/experiments/vision"
python breakhis_kl_benchmark.py \
    --methods baseline replace_mean patchcutout arch_mod temperature platt mcal_ce \
    --runs 10
echo "✓ BreakHis completed"
echo ""

# ======================================================================
# LANGUAGE MODALITY (2 datasets)
# ======================================================================

echo "======================================================================"
echo "LANGUAGE MODALITY"
echo "======================================================================"
echo ""

# 4. MedQA
echo "----------------------------------------------------------------------"
echo "[4/8] Running MedQA benchmark..."
echo "----------------------------------------------------------------------"
cd "${SCRIPT_DIR}/experiments/language"
python medqa_kl_benchmark.py \
    --methods baseline temperature platt mcal_ce token_drop attention_mask \
    --runs 10
echo "✓ MedQA completed"
echo ""

# 5. MedMCQA
echo "----------------------------------------------------------------------"
echo "[5/8] Running MedMCQA benchmark..."
echo "----------------------------------------------------------------------"
cd "${SCRIPT_DIR}/experiments/language"
python medmcqa_kl_benchmark.py \
    --methods baseline temperature platt mcal_ce token_drop attention_mask \
    --runs 10
echo "✓ MedMCQA completed"
echo ""

# ======================================================================
# TABULAR MODALITY (3 datasets)
# ======================================================================

echo "======================================================================"
echo "TABULAR MODALITY"
echo "======================================================================"
echo ""

# 6. PhysioNet
echo "----------------------------------------------------------------------"
echo "[6/8] Running PhysioNet benchmark..."
echo "----------------------------------------------------------------------"
cd "${SCRIPT_DIR}/experiments/tabular"
python physionet_kl_benchmark.py \
    --methods baseline replace_mean temperature platt mcal_ce arch_mod retrain \
    --runs 10
echo "✓ PhysioNet completed"
echo ""

# 7. Breast Cancer
echo "----------------------------------------------------------------------"
echo "[7/8] Running Breast Cancer benchmark..."
echo "----------------------------------------------------------------------"
cd "${SCRIPT_DIR}/experiments/tabular"
python breast_cancer_kl_benchmark.py \
    --methods baseline replace_mean temperature platt mcal_ce arch_mod retrain \
    --runs 10
echo "✓ Breast Cancer completed"
echo ""

# 8. Cardiotocography (CTG)
echo "----------------------------------------------------------------------"
echo "[8/8] Running Cardiotocography (CTG) benchmark..."
echo "----------------------------------------------------------------------"
cd "${SCRIPT_DIR}/experiments/tabular"
python ctg_kl_benchmark.py \
    --methods baseline replace_mean temperature platt mcal_ce arch_mod retrain \
    --runs 10
echo "✓ CTG completed"
echo ""

# ======================================================================
# COMPLETION SUMMARY
# ======================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "======================================================================"
echo "ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  - Vision datasets:   3/3 completed (MRI, CheXpert, BreakHis)"
echo "  - Language datasets: 2/2 completed (MedQA, MedMCQA)"
echo "  - Tabular datasets:  3/3 completed (PhysioNet, Breast Cancer, CTG)"
echo "  - Total datasets:    8/8"
echo "  - Runs per dataset:  10"
echo ""
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to respective results directories:"
echo "  - Vision:   experiments/vision/results/"
echo "  - Language: experiments/language/results/"
echo "  - Tabular:  experiments/tabular/results/"
echo ""
echo "======================================================================"
