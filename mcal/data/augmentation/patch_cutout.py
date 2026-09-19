"""PatchCutout data augmentation technique for Vision Transformers."""

import torch
import numpy as np
from typing import Union, Tuple
import logging

from .patch_drop import patch_segment, remove_random_features

logger = logging.getLogger(__name__)


class PatchCutout:
    """Randomly mask out patches from an image using patch-based segmentation.
    
    This is particularly useful for Vision Transformers where the image is 
    naturally divided into patches. The augmentation removes entire patches
    rather than arbitrary rectangular regions.
    
    Args:
        patch_height (int): Height of each patch in pixels
        patch_width (int): Width of each patch in pixels  
        removal_fraction (float): Fraction of patches to remove
        random_removal_fraction (bool): Whether to randomize removal fraction
        random_dist (str): Distribution for random removal ("binomial" or "uniform")
        fill_val (Union[int, float, Tuple]): Value to fill removed patches
    """
    
    def __init__(
        self,
        patch_height: int = 16,
        patch_width: int = 16,
        removal_fraction: float = 0.5,
        random_removal_fraction: bool = False,
        random_dist: str = "binomial",
        fill_val: Union[int, float, Tuple] = 0
    ):
        """Initialize PatchCutout augmentation.
        
        Args:
            patch_height: Height of each patch in pixels
            patch_width: Width of each patch in pixels
            removal_fraction: Fraction of patches to remove (0.0 to 1.0)
            random_removal_fraction: Whether to use random removal fraction
            random_dist: Distribution for random removal ("binomial" or "uniform") 
            fill_val: Value to fill removed patches (scalar or tuple for RGB)
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.removal_fraction = removal_fraction
        self.random_removal_fraction = random_removal_fraction
        self.random_dist = random_dist
        self.fill_val = fill_val
        
        if random_dist not in ["binomial", "uniform"]:
            raise ValueError("random_dist must be 'binomial' or 'uniform'")
        
        if not 0.0 <= removal_fraction <= 1.0:
            raise ValueError("removal_fraction must be between 0.0 and 1.0")
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply PatchCutout augmentation to image.
        
        Args:
            img: Input tensor image of size (C, H, W)
            
        Returns:
            Tensor: Image with random patches removed
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if len(img.shape) != 3:
            raise ValueError("Input tensor must have 3 dimensions (C, H, W)")
        
        channels, image_height, image_width = img.shape
        
        # Calculate removal fraction
        if self.random_removal_fraction:
            removal_fraction = self._calculate_random_removal_fraction(
                image_height, image_width
            )
        else:
            removal_fraction = self.removal_fraction
        
        # Apply patch removal
        # print("patchcutout*"*50)

        augmented_img = remove_random_features(
            image=img,
            segmentation_fn=patch_segment,
            removal_fraction=removal_fraction,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            fill_val=self.fill_val
        )
        # print("patchcutoutexit*"*50)

        
        return augmented_img
    
    def _calculate_random_removal_fraction(
        self, 
        image_height: int, 
        image_width: int
    ) -> float:
        """Calculate random removal fraction based on distribution."""
        # Calculate total number of patches
        n_patches_h = image_height // self.patch_height
        n_patches_w = image_width // self.patch_width
        total_patches = n_patches_h * n_patches_w
        
        if total_patches <= 1:
            return 0.0  # Can't remove patches if there's only one or none
        
        if self.random_dist == "binomial":
            # Binomial distribution with p=0.5
            p = 0.5
            n_removed = np.random.binomial(total_patches, p)
            removal_fraction = n_removed / total_patches
        
        elif self.random_dist == "uniform":
            # Uniform distribution over [0, total_patches]
            n_removed = np.random.randint(0, total_patches)
            removal_fraction = n_removed / total_patches
        
        else:
            raise ValueError(f"Unknown random distribution: {self.random_dist}")
        
        # Ensure we don't remove all patches (keep at least one)

        max_removal = (total_patches - 1) / total_patches
        removal_fraction = min(removal_fraction, max_removal)
        
        return removal_fraction
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"patch_height={self.patch_height}, "
                f"patch_width={self.patch_width}, "
                f"removal_fraction={self.removal_fraction}, "
                f"random_removal_fraction={self.random_removal_fraction}, "
                f"random_dist='{self.random_dist}', "
                f"fill_val={self.fill_val})")

