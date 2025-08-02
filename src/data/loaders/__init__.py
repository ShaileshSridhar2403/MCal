"""Data loaders for different modalities and datasets."""

from .base_loader import BaseDataLoader, VisionDataLoader
from .vision_loaders import (
    BreakHisLoader,
    MRILoader,
    ChexPertLoader,
    ImageNetLoader,
    ImageNetteLoader,
    get_vision_loader,
    VISION_LOADERS
)
from .language_loaders import (
    MedMCQALoader,
    MedQALoader,
    AI2ARCLoader,
    PhysionetLoader as LanguagePhysionetLoader,
    get_language_loader,
    LANGUAGE_LOADERS
)
from .tabular_loaders import (
    WisconsinBreastCancerLoader,
    PhysionetLoader,
    CardiotocographyLoader,
    get_tabular_loader,
    TABULAR_LOADERS
)

# Unified loader registry
ALL_LOADERS = {
    **VISION_LOADERS,
    **LANGUAGE_LOADERS,
    **TABULAR_LOADERS
}


def get_loader(dataset_name: str, modality: str = None, **kwargs):
    """Get a dataset loader by name and optional modality.
    
    Args:
        dataset_name: Name of the dataset
        modality: Modality type ('vision', 'language', 'tabular')
        **kwargs: Arguments passed to loader constructor
        
    Returns:
        Dataset loader instance
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    if modality is not None:
        if modality == "vision":
            return get_vision_loader(dataset_name, **kwargs)
        elif modality == "language":
            return get_language_loader(dataset_name, **kwargs)
        elif modality == "tabular":
            return get_tabular_loader(dataset_name, **kwargs)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    # Try to find in all loaders
    if dataset_name in ALL_LOADERS:
        return ALL_LOADERS[dataset_name](**kwargs)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(ALL_LOADERS.keys())}"
        )


def list_datasets(modality: str = None) -> list:
    """List available datasets.
    
    Args:
        modality: Optional modality filter
        
    Returns:
        List of available dataset names
    """
    if modality is None:
        return list(ALL_LOADERS.keys())
    elif modality == "vision":
        return list(VISION_LOADERS.keys())
    elif modality == "language":
        return list(LANGUAGE_LOADERS.keys())
    elif modality == "tabular":
        return list(TABULAR_LOADERS.keys())
    else:
        raise ValueError(f"Unknown modality: {modality}")


__all__ = [
    # Base classes
    "BaseDataLoader",
    "VisionDataLoader",
    
    # Vision loaders
    "BreakHisLoader",
    "MRILoader", 
    "ChexPertLoader",
    "ImageNetLoader",
    "ImageNetteLoader",
    "get_vision_loader",
    
    # Language loaders
    "MedMCQALoader",
    "MedQALoader",
    "AI2ARCLoader",
    "LanguagePhysionetLoader",
    "get_language_loader",
    
    # Tabular loaders
    "WisconsinBreastCancerLoader",
    "PhysionetLoader",
    "CardiotocographyLoader", 
    "get_tabular_loader",
    
    # Utilities
    "get_loader",
    "list_datasets",
    "ALL_LOADERS",
    "VISION_LOADERS",
    "LANGUAGE_LOADERS",
    "TABULAR_LOADERS",
]