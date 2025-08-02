"""Vision models for image classification tasks."""

from .vit_model import (
    VisionTransformer,
    ViTWithPatchDrop,
    PatchEmbedding,
    MultiHeadSelfAttention,
    TransformerBlock,
    create_vit_small,
    create_vit_base,
    create_vit_large,
    get_vit_model,
    VIT_MODELS
)
from .cnn_model import (
    BasicCNN,
    ResNetWrapper,
    EfficientNetWrapper,
    CustomCNN,
    get_cnn_model,
    CNN_MODELS
)

# Combined model registry
VISION_MODELS = {
    **VIT_MODELS,
    **CNN_MODELS
}


def get_vision_model(model_name: str, num_classes: int, **kwargs):
    """Get a vision model by name.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        Vision model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name in VIT_MODELS:
        return get_vit_model(model_name, num_classes, **kwargs)
    elif model_name in CNN_MODELS:
        return get_cnn_model(model_name, num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown vision model: {model_name}. "
            f"Available models: {list(VISION_MODELS.keys())}"
        )


def list_vision_models() -> list:
    """List all available vision models."""
    return list(VISION_MODELS.keys())


__all__ = [
    # ViT models
    "VisionTransformer",
    "ViTWithPatchDrop",
    "PatchEmbedding",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "create_vit_small",
    "create_vit_base", 
    "create_vit_large",
    "get_vit_model",
    
    # CNN models
    "BasicCNN",
    "ResNetWrapper",
    "EfficientNetWrapper",
    "CustomCNN",
    "get_cnn_model",
    
    # Utilities
    "get_vision_model",
    "list_vision_models",
    "VISION_MODELS",
    "VIT_MODELS",
    "CNN_MODELS",
]