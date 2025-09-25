"""
MCal Tabular Benchmarks

This module provides benchmarking capabilities for tabular data using various
calibration methods including MCal, Platt scaling, temperature scaling, and
LogitsSharp transforms.

Supported datasets:
- PhysioNet Challenge 2012 (ICU mortality prediction) - Binary classification
- Cardiotocography (CTG) - Multi-class fetal monitoring classification

Supported models:
- XGBoost with various imputation strategies
"""

__version__ = "1.0.0"
__author__ = "MCal Team"

# PhysioNet benchmarks (binary classification)
try:
    from .physionet_data_setup import load_physionet_data
    from .physionet_kl_benchmark import process_physionet_dataset
except ImportError:
    pass

# CTG benchmarks (multi-class classification)
try:
    from .ctg_data_setup import load_ctg_data
    from .ctg_kl_benchmark import process_ctg_dataset
except ImportError:
    pass