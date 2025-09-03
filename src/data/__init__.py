"""Data loading and preprocessing utilities."""

from .loaders import (
    # Base classes
    BaseDataLoader,
    VisionDataLoader,
    
    # Loader functions
    get_loader,
    get_vision_loader,
    get_language_loader,
    get_tabular_loader,
    list_datasets,
    
    # Specific loaders
    BreakHisLoader,
    MRILoader,
    ChexPertLoader,
    ImageNetLoader,
    ImageNetteLoader,
    MedMCQALoader,
    MedQALoader,
    AI2ARCLoader,
    WisconsinBreastCancerLoader,
    PhysionetLoader,
    CardiotocographyLoader,
    
    # Registries
    ALL_LOADERS,
    VISION_LOADERS,
    LANGUAGE_LOADERS,
    TABULAR_LOADERS,
)

from .augmentation import (
    # Cutout variants
    Cutout,
    RandomCutout,
    AdaptiveCutout,
    
    # PatchCutout variants
    PatchCutout,
    RandomPatchCutout,
    AdaptivePatchCutout,
    GridPatchCutout,
    
    # Patch utilities
    patch_segment,
    remove_random_features,
    remove_mask,
    # get_patch_indices,  
    # get_total_patches,
    # create_random_patch_mask,
)

__all__ = [
    # Data loaders
    "BaseDataLoader",
    "VisionDataLoader",
    "get_loader",
    "get_vision_loader", 
    "get_language_loader",
    "get_tabular_loader",
    "list_datasets",
    "BreakHisLoader",
    "MRILoader",
    "ChexPertLoader", 
    "ImageNetLoader",
    "ImageNetteLoader",
    "MedMCQALoader",
    "MedQALoader",
    "AI2ARCLoader",
    "WisconsinBreastCancerLoader",
    "PhysionetLoader",
    "CardiotocographyLoader",
    "ALL_LOADERS",
    "VISION_LOADERS",
    "LANGUAGE_LOADERS", 
    "TABULAR_LOADERS",
    
    # Augmentation
    "Cutout",
    "RandomCutout",
    "AdaptiveCutout",
    "PatchCutout",
    "RandomPatchCutout",
    "AdaptivePatchCutout",
    "GridPatchCutout",
    "patch_segment",
    "remove_random_features",
    "remove_mask",
    # "get_patch_indices",
    # "get_total_patches",
    # "create_random_patch_mask",
]