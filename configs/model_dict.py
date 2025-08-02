"""Model dictionary mapping datasets to trained model files.

This file maps each dataset and augmentation type to the corresponding
pre-trained model file. Models are expected in the saved_models/ directory.
"""

from pathlib import Path
import os

# Base directory for saved models
SAVED_MODELS_DIR = Path(__file__).parent.parent / "saved_models"

# Model dictionary mapping dataset -> augmentation -> model file
MODEL_DICT = {
    "mri": {
        "vanilla": "vit_timm_standard_mri_ps64_35e.pth",
        "PatchCutout": "vit_timm_patchcutoutbinom_mri_ps64_55e.pth"
    },
    "imagenette": {
        "vanilla": "vit_timm_vanilla_imagenette_10e_100pt0tr99pt9te.pth",
        "PatchCutout": "vit_timm_randombinompatchcutout_imagenette_ps64_70e_94pt8tr99pt9te.pth"
    },
    "chexpert": {
        "vanilla": "vit_timm_vanilla_chexpert_ps64_85tr76te.pth",
        "PatchCutout": "vit_timm_patchcutout_chexpert_ps64_83tr75te.pth"
    },
    "breakhis": {
        "vanilla": "breakhis_vanilla_finetuned.pth",
        "PatchCutout": "vit_timm_PatchCutout_breakhis_ps56_98tr85te.pth"
    },
    "imagenet": {
        "vanilla": "imagenet_dummy"  # Placeholder
    }
}

# Legacy alias for compatibility
model_dict = MODEL_DICT


def get_model_path(dataset: str, augmentation: str = "vanilla") -> Path:
    """Get the full path to a model file.
    
    Args:
        dataset: Dataset name (e.g., 'mri', 'breakhis')
        augmentation: Augmentation type (e.g., 'vanilla', 'PatchCutout')
        
    Returns:
        Path to the model file
        
    Raises:
        KeyError: If dataset or augmentation not found
        FileNotFoundError: If model file doesn't exist
    """
    if dataset not in MODEL_DICT:
        raise KeyError(f"Unknown dataset: {dataset}. Available: {list(MODEL_DICT.keys())}")
    
    if augmentation not in MODEL_DICT[dataset]:
        available_augs = list(MODEL_DICT[dataset].keys())
        raise KeyError(f"Unknown augmentation '{augmentation}' for dataset '{dataset}'. Available: {available_augs}")
    
    model_filename = MODEL_DICT[dataset][augmentation]
    model_path = SAVED_MODELS_DIR / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model_path


def list_available_models() -> dict:
    """List all available models.
    
    Returns:
        Dictionary of available models by dataset and augmentation
    """
    available = {}
    
    for dataset, augs in MODEL_DICT.items():
        available[dataset] = {}
        for aug, filename in augs.items():
            model_path = SAVED_MODELS_DIR / filename
            available[dataset][aug] = {
                "filename": filename,
                "exists": model_path.exists(),
                "path": str(model_path)
            }
    
    return available


def check_model_availability() -> None:
    """Check which models are available and print status."""
    available = list_available_models()
    
    print("Model Availability Status:")
    print("=" * 50)
    
    for dataset, augs in available.items():
        print(f"\n{dataset.upper()}:")
        for aug, info in augs.items():
            status = "✓" if info["exists"] else "✗"
            print(f"  {aug:15} {status} {info['filename']}")
    
    # Summary
    total_models = sum(len(augs) for augs in available.values())
    existing_models = sum(
        sum(1 for info in augs.values() if info["exists"]) 
        for augs in available.values()
    )
    
    print(f"\nSummary: {existing_models}/{total_models} models available")


if __name__ == "__main__":
    check_model_availability()