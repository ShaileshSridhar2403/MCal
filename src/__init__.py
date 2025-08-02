"""MCal - Model Calibration Framework

A comprehensive framework for model calibration across vision, language, and tabular modalities.
"""

__version__ = "0.1.0"
__author__ = "MCal Team"

from . import calibrators
from . import transforms
from . import data
from . import models
from . import evaluation
from . import utils

# Import configs and model loading for easy access
try:
    import sys
    sys.path.append(str(__file__).replace('/src/__init__.py', ''))
    from configs import (
        MODEL_DICT,
        DATASET_CONFIGS,
        get_model_path,
        get_dataset_config
    )
    from .models import load_model_for_dataset, load_expectation_tracking_model
    
    __all__ = [
        "calibrators",
        "transforms", 
        "data",
        "models",
        "evaluation",
        "utils",
        
        # Configs and model loading
        "MODEL_DICT",
        "DATASET_CONFIGS", 
        "get_model_path",
        "get_dataset_config",
        "load_model_for_dataset",
        "load_expectation_tracking_model",
    ]
except ImportError:
    # Configs not available, use basic exports
    __all__ = [
        "calibrators",
        "transforms", 
        "data",
        "models",
        "evaluation",
        "utils"
    ]