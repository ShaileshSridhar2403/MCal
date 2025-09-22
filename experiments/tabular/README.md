# MCal Tabular Benchmarks

This module provides tabular data calibration benchmarks for the MCal framework, focusing on XGBoost models with missing data robustness.

## Features

- **Dataset Support**: PhysioNet Challenge 2012 (ICU mortality prediction)
- **Model Support**: XGBoost with multiple imputation strategies
- **Missing Data Simulation**: Fractionwise feature ablation (0-90% removal)
- **Calibration Methods**: MCal, MCal_CE, Platt Scaling, Temperature Scaling, LogitsSharp
- **MCal Integration**: Follows identical patterns to vision and language benchmarks

## Quick Start

### Data Preprocessing (First Time Setup)

```bash
# Activate MCal environment
source /home/antonxue/shailesh/MCal/mcal/bin/activate

# Process PhysioNet data (if you have PhysionetChallenge2012-set-a.csv.gz)
cd /home/antonxue/shailesh/MCal/experiments/tabular
python run_physionet_processing.py

# Or process with explicit path:
python process_physionet_data.py --input_path /path/to/PhysionetChallenge2012-set-a.csv.gz
```

### Basic Usage

```bash
# Run demo with synthetic data (works without PhysioNet)
python demo_tabular_benchmark.py

# Run basic benchmark (requires PhysioNet data)
python tabular_kl_benchmark.py --methods baseline mcal platt temperature
```

### Full Benchmark

```bash
# Run complete benchmark with all methods
python tabular_kl_benchmark.py \
    --methods baseline mcal mcal_ce platt temperature logits_sharp \
    --runs 3 \
    --samples 1000 \
    --fractions 10 \
    --device cuda \
    --save_dir ./results \
    --imputation_strategy mean \
    --missingness_range 0-30
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--methods` | `['baseline', 'mcal', 'platt', 'temperature', 'logits_sharp']` | Calibration methods to evaluate |
| `--runs` | `3` | Number of independent runs |
| `--samples` | `1000` | Number of samples per run |
| `--fractions` | `10` | Number of missing data fractions |
| `--device` | `"cuda"` | Computing device (cuda/cpu) |
| `--save_dir` | `"./results"` | Results output directory |
| `--imputation_strategy` | `"mean"` | Model imputation strategy (mean/zero/xgboost_native) |
| `--missingness_range` | `"0-30"` | PhysioNet missingness range (0-30/30-100/full) |

## Data Processing Scripts

### PhysioNet Preprocessing
- **`process_physionet_data.py`** - Complete preprocessing pipeline using exact XAI_Benchmark logic
- **`run_physionet_processing.py`** - Quick processing script with auto-detection of data files

The preprocessing scripts convert raw PhysioNet Challenge 2012 data into the required missingness-level format:
```bash
# Auto-detect and process PhysioNet data
python run_physionet_processing.py

# Process with explicit path
python process_physionet_data.py --input_path PhysionetChallenge2012-set-a.csv.gz
```

This creates files like:
```
missingness_levels/
├── missingness_000_010.csv.gz  # 0-10% missing data
├── missingness_010_020.csv.gz  # 10-20% missing data
├── missingness_020_030.csv.gz  # 20-30% missing data
└── ... (up to 90-100%)
```

## Available Methods

### Calibration Methods
- **baseline**: Raw XGBoost predictions (no calibration)
- **mcal**: MCal vector scaling with uniform target distribution
- **mcal_ce**: MCal with cross-entropy loss and target labels
- **platt**: Platt scaling (logistic regression post-processing)
- **temperature**: Temperature scaling parameter optimization
- **logits_sharp**: LogitsSharp transform (from XAI_Benchmark)

### Imputation Strategies
- **mean**: Replace missing values with feature means
- **zero**: Replace missing values with zeros
- **xgboost_native**: Use XGBoost's built-in missing value handling

## Output Files

After running the benchmark, the following files are generated:

```
results/
├── json/
│   └── aggregated_results_physionet.json     # Detailed numerical results
├── kl_comparison_table_physionet.txt         # Formatted comparison table
└── kl_divergence_physionet.png              # Visualization plots
```

### Example Results Table

```
+---------------------------+-------------------------+-------------------------+
| Method                    | Average KL (Prob)      | Average KL (Argmax)    |
+===========================+=========================+=========================+
| Original                  | 2.15e-02 ± 3.21e-03    | 4.67e-02 ± 5.12e-03    |
+---------------------------+-------------------------+-------------------------+
| MCal (Vector Scaling)     | 1.89e-02 ± 2.87e-03    | 3.98e-02 ± 4.33e-03    |
+---------------------------+-------------------------+-------------------------+
| MCal_CE (Cross-Entropy)   | 1.76e-02 ± 2.65e-03    | 3.71e-02 ± 4.01e-03    |
+---------------------------+-------------------------+-------------------------+
| Platt Scaling             | 1.92e-02 ± 2.91e-03    | 4.05e-02 ± 4.44e-03    |
+---------------------------+-------------------------+-------------------------+
| Temperature Scaling       | 1.88e-02 ± 2.84e-03    | 3.94e-02 ± 4.28e-03    |
+---------------------------+-------------------------+-------------------------+
| LogitsSharp Transform     | 1.71e-02 ± 2.58e-03    | 3.62e-02 ± 3.87e-03    |
+---------------------------+-------------------------+-------------------------+
```

## Testing

Run the test suite to verify installation:

```bash
python test_tabular_benchmark.py
```

The test suite includes:
- Import verification
- Missing data simulation
- KL divergence calculation
- XGBoost predictor functionality
- Calibration method application
- End-to-end benchmark structure

## Implementation Details

### Architecture

The tabular benchmark follows MCal's established patterns:

1. **Data Loading**: `physionet_data_setup.py` handles dataset preparation
2. **Model Training**: `xgboost_utils.py` provides XGBoost integration
3. **Utilities**: `tabular_utils.py` contains common helper functions
4. **Main Benchmark**: `tabular_kl_benchmark.py` orchestrates the evaluation

### Missing Data Simulation

- **Strategy**: Random feature ablation per sample
- **Fractions**: Linear progression from 0% to 90% missing features
- **Preprocessing**: Multiple imputation strategies supported
- **Evaluation**: KL divergence against clean distribution (0% missing)

### Calibration Integration

All calibration methods use identical interfaces to vision/language benchmarks:

```python
def apply_transform(outputs, labels, method, device=None, **kwargs):
    """Apply calibration transform following MCal pattern."""
    if method == 'mcal':
        return apply_mcal_calibrator(outputs, device, **kwargs)
    # ... other methods
```

## Requirements

- **Python**: 3.8+
- **Core**: NumPy, Pandas, Scikit-learn
- **ML**: XGBoost 3.0+, PyTorch
- **MCal**: All existing MCal dependencies
- **Optional**: XAI_Benchmark (for LogitsSharp transform)

## Integration with MCal

The tabular benchmarks seamlessly integrate with MCal's ecosystem:

- **Consistent API**: Same function signatures as vision/language benchmarks
- **Unified Results**: Compatible JSON output format and aggregation
- **Shared Calibrators**: Uses existing MCal calibration modules
- **Common Utilities**: Leverages MCal's optimization and evaluation tools

## Future Extensions

The current implementation provides a foundation for:

1. **Additional Datasets**: Cardiotocography, Breast Cancer Wisconsin
2. **More Models**: TabPFN, Random Forest, Neural Networks
3. **Advanced Missing Data**: MCAR, MAR, MNAR simulation patterns
4. **Explainability**: SHAP/LIME integration
5. **Uncertainty**: Ensemble methods and uncertainty quantification

## Troubleshooting

### Common Issues

1. **Missing XGBoost**: Install with `pip install xgboost`
2. **CUDA Issues**: Use `--device cpu` for CPU-only execution
3. **Memory Errors**: Reduce `--samples` and `--fractions` for testing
4. **Import Errors**: Ensure MCal virtual environment is activated

### Performance Tips

- Use `--device cuda` for GPU acceleration of calibration
- Start with smaller `--samples` (100-500) for initial testing
- Use `--runs 1` for quick verification
- Enable `--imputation_strategy xgboost_native` for best missing data handling

---

For more information, see the [implementation plan](../../tabular_benchmark_implementation_plan.md) and [MCal documentation](../../README.md).