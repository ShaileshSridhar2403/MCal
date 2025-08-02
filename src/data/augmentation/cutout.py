"""Cutout data augmentation technique."""

import torch
import numpy as np
from typing import Union


class Cutout:
    """Randomly mask out one or more patches from an image.
    
    This implementation is based on the original Cutout paper:
    "Improved Regularization of Convolutional Neural Networks with Cutout"
    https://arxiv.org/abs/1708.04552
    
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        fill_value (Union[int, float]): Value to fill the cutout regions.
    """
    
    def __init__(
        self, 
        n_holes: int = 1, 
        length: int = 16,
        fill_value: Union[int, float] = 0
    ):
        """Initialize Cutout augmentation.
        
        Args:
            n_holes: Number of patches to cut out of each image
            length: The length (in pixels) of each square patch
            fill_value: Value to fill the cutout regions
        """
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply cutout augmentation to image.
        
        Args:
            img: Tensor image of size (C, H, W).
            
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if len(img.shape) != 3:
            raise ValueError("Input tensor must have 3 dimensions (C, H, W)")
        
        # Get image dimensions
        c, h, w = img.shape
        
        # Create a copy to avoid modifying the original
        img_augmented = img.clone()
        
        # Create mask
        mask = torch.ones((h, w), dtype=torch.float32, device=img.device)
        
        for _ in range(self.n_holes):
            # Random center point
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            
            # Calculate cutout boundaries
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            # Apply mask
            mask[y1:y2, x1:x2] = 0.0
        
        # Expand mask to match image channels
        mask = mask.unsqueeze(0).expand(c, -1, -1)
        
        # Apply cutout
        img_augmented = img_augmented * mask
        
        # Fill cutout regions with specified value
        if self.fill_value != 0:
            fill_mask = (mask == 0)
            img_augmented[fill_mask] = self.fill_value
        
        return img_augmented
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length}, fill_value={self.fill_value})"


class RandomCutout(Cutout):
    """Cutout with random number of holes and lengths."""
    
    def __init__(
        self,
        n_holes_range: tuple = (1, 3),
        length_range: tuple = (8, 32),
        fill_value: Union[int, float] = 0
    ):
        """Initialize RandomCutout.
        
        Args:
            n_holes_range: Range of number of holes (min, max)
            length_range: Range of hole lengths (min, max)
            fill_value: Value to fill the cutout regions
        """
        # Initialize with default values (will be overridden in __call__)
        super().__init__(n_holes=1, length=16, fill_value=fill_value)
        
        self.n_holes_range = n_holes_range
        self.length_range = length_range
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random cutout augmentation."""
        # Randomly sample parameters
        self.n_holes = torch.randint(
            self.n_holes_range[0], 
            self.n_holes_range[1] + 1, 
            (1,)
        ).item()
        
        self.length = torch.randint(
            self.length_range[0],
            self.length_range[1] + 1,
            (1,)
        ).item()
        
        # Apply cutout with random parameters
        return super().__call__(img)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(n_holes_range={self.n_holes_range}, "
                f"length_range={self.length_range}, fill_value={self.fill_value})")


class AdaptiveCutout(Cutout):
    """Cutout that adapts hole size based on image size."""
    
    def __init__(
        self,
        n_holes: int = 1,
        length_ratio: float = 0.1,
        fill_value: Union[int, float] = 0
    ):
        """Initialize AdaptiveCutout.
        
        Args:
            n_holes: Number of holes to cut out
            length_ratio: Ratio of image size to use as hole length
            fill_value: Value to fill the cutout regions
        """
        super().__init__(n_holes=n_holes, length=1, fill_value=fill_value)
        self.length_ratio = length_ratio
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply adaptive cutout augmentation."""
        # Calculate adaptive length based on image size
        _, h, w = img.shape
        min_dim = min(h, w)
        self.length = int(min_dim * self.length_ratio)
        
        # Ensure minimum length of 1
        self.length = max(1, self.length)
        
        return super().__call__(img)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(n_holes={self.n_holes}, "
                f"length_ratio={self.length_ratio}, fill_value={self.fill_value})")