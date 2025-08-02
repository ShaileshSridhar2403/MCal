"""Data augmentation techniques for different modalities."""

from .cutout import Cutout, RandomCutout, AdaptiveCutout
from .patch_cutout import (
    PatchCutout, 
    RandomPatchCutout, 
    AdaptivePatchCutout,
    GridPatchCutout
)
from .patch_drop import (
    patch_segment, 
    remove_random_features, 
    remove_mask,
    get_patch_indices,
    get_total_patches,
    create_random_patch_mask
)

__all__ = [
    # Cutout variants
    "Cutout",
    "RandomCutout",
    "AdaptiveCutout",
    
    # PatchCutout variants
    "PatchCutout",
    "RandomPatchCutout", 
    "AdaptivePatchCutout",
    "GridPatchCutout",
    
    # Patch utilities
    "patch_segment",
    "remove_random_features",
    "remove_mask",
    "get_patch_indices",
    "get_total_patches", 
    "create_random_patch_mask",
]