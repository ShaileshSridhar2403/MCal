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

