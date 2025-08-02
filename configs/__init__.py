"""Configuration files for datasets, models, and training."""

from .model_dict import (
    MODEL_DICT,
    model_dict,  # Legacy alias
    get_model_path,
    list_available_models,
    check_model_availability,
    SAVED_MODELS_DIR
)

from .dataset_configs import (
    DATASET_CONFIGS,
    VIT_CONFIGS,
    TRAINING_CONFIGS,
    BASE_CONFIG,
    get_dataset_config,
    get_vit_config,
    get_training_config,
    get_combined_config,
    print_config_summary,
    # Legacy aliases
    breakhis_config,
    mri_config
)

__all__ = [
    # Model dictionary
    "MODEL_DICT",
    "model_dict",
    "get_model_path",
    "list_available_models", 
    "check_model_availability",
    "SAVED_MODELS_DIR",
    
    # Dataset configs
    "DATASET_CONFIGS",
    "VIT_CONFIGS",
    "TRAINING_CONFIGS",
    "BASE_CONFIG",
    "get_dataset_config",
    "get_vit_config",
    "get_training_config",
    "get_combined_config",
    "print_config_summary",
    
    # Legacy aliases
    "breakhis_config",
    "mri_config",
]