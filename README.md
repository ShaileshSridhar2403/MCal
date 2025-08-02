# MCal - Model Calibration Framework

A comprehensive framework for model calibration across vision, language, and tabular modalities.

## Directory Structure

```
MCal/
├── src/                    # Core library code
│   ├── calibrators/        # Calibration algorithms
│   ├── transforms/         # Probability transformation pipeline
│   ├── data/              # Dataset handling utilities
│   ├── models/            # Model wrappers and training
│   ├── evaluation/        # Metrics and evaluation
│   └── utils/             # General utilities
├── experiments/           # Research experiments and benchmarks
├── tests/                # Test suite
├── configs/              # Configuration files
├── finetune/             # Model fine-tuning scripts
└── notebooks/            # Example usage and analysis
```

## Key Features

- **Multi-modal Support**: Vision, language, and tabular data
- **Extensible Architecture**: Base classes for easy extension
- **Comprehensive Calibration**: Multiple calibration techniques (MCal, Platt, Temperature)
- **Transformation Pipeline**: Sophisticated probability transformation framework
- **Research-Ready**: Experiment infrastructure with result storage

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from mcal.calibrators import MCal
from mcal.data.loaders import load_vision_dataset

# Load data and model predictions
data = load_vision_dataset('breakhis')
ablated_probs, clean_probs = get_predictions(data)

# Calibrate
calibrator = MCal(num_classes=8)
calibrator.fit(ablated_probs, clean_probs)
calibrated_probs = calibrator(ablated_probs)
```