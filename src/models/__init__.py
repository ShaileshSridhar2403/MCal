"""Model implementations for different modalities."""

from .base_model import (
    BaseModel,
    ExpectationTrackingModel,
    ModelWrapper,
    EnsembleModel
)
from .model_loader import (
    ModelLoader,
    load_model_for_dataset,
    load_expectation_tracking_model,
    analyze_saved_models
)
from .vision import (
    VisionTransformer,
    ViTWithPatchDrop,
    BasicCNN,
    ResNetWrapper,
    EfficientNetWrapper,
    CustomCNN,
    get_vision_model,
    list_vision_models,
    VISION_MODELS
)
from .language import (
    LLMWrapper,
    LlamaWrapper,
    FalconWrapper,
    MistralWrapper,
    get_language_model,
    LANGUAGE_MODELS
)
from .tabular import (
    MLPClassifier,
    XGBoostWrapper,
    TabTransformerWrapper,
    TabPFNWrapper,
    get_tabular_model,
    TABULAR_MODELS
)

# Combined model registry
ALL_MODELS = {
    **VISION_MODELS,
    **LANGUAGE_MODELS,
    **TABULAR_MODELS
}


def get_model(model_name: str, modality: str = None, **kwargs):
    """Get a model by name and optional modality.
    
    Args:
        model_name: Name of the model
        modality: Modality type ('vision', 'language', 'tabular')
        **kwargs: Additional arguments
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if modality is not None:
        if modality == "vision":
            return get_vision_model(model_name, **kwargs)
        elif modality == "language":
            return get_language_model(model_name, **kwargs)
        elif modality == "tabular":
            return get_tabular_model(model_name, **kwargs)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    # Try to find in all models
    if model_name in VISION_MODELS:
        return get_vision_model(model_name, **kwargs)
    elif model_name in LANGUAGE_MODELS:
        return get_language_model(model_name, **kwargs)
    elif model_name in TABULAR_MODELS:
        return get_tabular_model(model_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(ALL_MODELS.keys())}"
        )


def list_models(modality: str = None) -> list:
    """List available models.
    
    Args:
        modality: Optional modality filter
        
    Returns:
        List of available model names
    """
    if modality is None:
        return list(ALL_MODELS.keys())
    elif modality == "vision":
        return list(VISION_MODELS.keys())
    elif modality == "language":
        return list(LANGUAGE_MODELS.keys())
    elif modality == "tabular":
        return list(TABULAR_MODELS.keys())
    else:
        raise ValueError(f"Unknown modality: {modality}")


__all__ = [
    # Base classes
    "BaseModel",
    "ExpectationTrackingModel",
    "ModelWrapper",
    "EnsembleModel",
    
    # Model loading
    "ModelLoader",
    "load_model_for_dataset",
    "load_expectation_tracking_model",
    "analyze_saved_models",
    
    # Vision models
    "VisionTransformer",
    "ViTWithPatchDrop",
    "BasicCNN",
    "ResNetWrapper",
    "EfficientNetWrapper",
    "CustomCNN",
    "get_vision_model",
    "list_vision_models",
    
    # Language models
    "LLMWrapper",
    "LlamaWrapper",
    "FalconWrapper",
    "MistralWrapper",
    "get_language_model",
    
    # Tabular models
    "MLPClassifier",
    "XGBoostWrapper",
    "TabTransformerWrapper",
    "TabPFNWrapper",
    "get_tabular_model",
    
    # Utilities
    "get_model",
    "list_models",
    "ALL_MODELS",
    "VISION_MODELS",
    "LANGUAGE_MODELS",
    "TABULAR_MODELS",
]