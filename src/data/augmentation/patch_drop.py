"""Patch-level operations for image augmentation."""

import torch
import numpy as np
import random
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def patch_segment(
    image: torch.Tensor,
    patch_height: int = 8,
    patch_width: int = 8,
    permute: Optional[Tuple[int, int, int]] = None,
    dtype: str = "torch"
) -> Union[torch.Tensor, np.ndarray]:
    """Segment an image into patches and return patch indices.
    
    Args:
        image: Input image tensor of shape (C, H, W)
        patch_height: Height of each patch
        patch_width: Width of each patch  
        permute: Optional permutation of dimensions
        dtype: Return type ("torch" or "numpy")
        
    Returns:
        Tensor/array of patch indices for each pixel
        
    Raises:
        ValueError: If image dimensions are not divisible by patch dimensions
    """
    if permute is not None:
        if isinstance(image, torch.Tensor):
            image = image.permute(*permute)
        else:
            image = np.transpose(image, permute)
    
    if len(image.shape) == 3:
        channels, image_height, image_width = image.shape
    elif len(image.shape) == 2:
        image_height, image_width = image.shape
    else:
        raise ValueError("Image must be 2D or 3D tensor")
    
    # Check if dimensions are divisible
    if image_height % patch_height != 0 or image_width % patch_width != 0:
        logger.warning(
            f"Image dimensions ({image_height}, {image_width}) not perfectly divisible by "
            f"patch dimensions ({patch_height}, {patch_width}). This may cause issues."
        )
    
    # Create patch indices
    if isinstance(image, torch.Tensor):
        row_indices = torch.arange(image_height, device=image.device)
        column_indices = torch.arange(image_width, device=image.device)
    else:
        row_indices = np.arange(image_height)
        column_indices = np.arange(image_width)
    
    # Calculate patch factors
    row_factors = row_indices // patch_height
    column_factors = column_indices // patch_width
    
    # Create factor matrices
    if isinstance(image, torch.Tensor):
        row_factor_matrix = row_factors[:, None]
        column_factor_matrix = column_factors[None, :]
    else:
        row_factor_matrix = row_factors[:, np.newaxis]
        column_factor_matrix = column_factors[np.newaxis, :]
    
    # Calculate segment indices
    segments = column_factor_matrix * (image_height // patch_height) + row_factor_matrix
    
    if dtype == "torch":
        if isinstance(segments, np.ndarray):
            segments = torch.from_numpy(segments)
        return segments.int()
    elif dtype == "numpy":
        if isinstance(segments, torch.Tensor):
            segments = segments.numpy()
        return segments
    else:
        raise ValueError("dtype must be 'torch' or 'numpy'")


def remove_random_features(
    image: torch.Tensor,
    segmentation_fn: callable,
    removal_fraction: float,
    patch_height: int,
    patch_width: int,
    fill_val: Union[int, float, Tuple] = 0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """Remove random patches from an image.
    
    Args:
        image: Input image tensor of shape (C, H, W)
        segmentation_fn: Function to segment image into patches
        removal_fraction: Fraction of patches to remove (0.0 to 1.0)
        patch_height: Height of each patch
        patch_width: Width of each patch
        fill_val: Value to fill removed patches
        seed: Random seed for reproducibility
        
    Returns:
        Image with random patches removed
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Get patch segments
    segments = segmentation_fn(image, patch_height, patch_width)
    
    # Calculate number of segments to retain
    min_segment = segments.min().item()
    max_segment = segments.max().item()
    total_segments = max_segment - min_segment + 1
    n_to_retain = int((1 - removal_fraction) * total_segments)
    
    # Ensure at least one segment is retained
    n_to_retain = max(1, n_to_retain)
    
    # Randomly select segments to retain
    all_segments = list(range(min_segment, max_segment + 1))
    retained_segments = random.sample(all_segments, n_to_retain)
    retained_features = torch.tensor(retained_segments, device=image.device)
    
    # Create mask for retained features
    expanded_retained_features = retained_features.view(1, -1, 1, 1)
    mask = (segments.unsqueeze(0) == expanded_retained_features)
    mask = mask.sum(1) > 0  # Combine all retained segments
    
    # Apply mask to image
    if len(image.shape) == 3:
        mask = mask.unsqueeze(0).expand_as(image)
        masked_image = image * mask.float()
        
        # Fill removed regions
        if fill_val != 0:
            if isinstance(fill_val, (tuple, list)):
                # Different fill value per channel
                fill_tensor = torch.tensor(fill_val, device=image.device).view(-1, 1, 1)
            else:
                # Same fill value for all channels
                fill_tensor = torch.tensor(fill_val, device=image.device)
            
            masked_image = masked_image + (1 - mask.float()) * fill_tensor
    else:
        # 2D image
        masked_image = image * mask.float()
        if fill_val != 0:
            masked_image = masked_image + (1 - mask.float()) * fill_val
    
    return masked_image


def remove_mask(
    image: torch.Tensor,
    segmentation_fn: callable,
    mask_vector: torch.Tensor,
    patch_height: int,
    patch_width: int,
    fill_val: Union[int, float, Tuple] = 0
) -> torch.Tensor:
    """Remove patches specified by a mask vector.
    
    Args:
        image: Input image tensor of shape (C, H, W)
        segmentation_fn: Function to segment image into patches
        mask_vector: Binary mask indicating which patches to remove (1 = remove, 0 = keep)
        patch_height: Height of each patch
        patch_width: Width of each patch
        fill_val: Value to fill removed patches
        
    Returns:
        Image with specified patches removed
    """
    # Get patch segments
    segments = segmentation_fn(image, patch_height, patch_width)
    
    # Find segments to retain (where mask is 0)
    retained_indices = torch.where(mask_vector == 0)[0]
    
    if len(retained_indices) == 0:
        logger.warning("All patches are masked out, returning filled image")
        if isinstance(fill_val, (tuple, list)):
            fill_tensor = torch.tensor(fill_val, device=image.device).view(-1, 1, 1)
            return torch.ones_like(image) * fill_tensor
        else:
            return torch.ones_like(image) * fill_val
    
    # Create mask for retained features
    expanded_retained_features = retained_indices.view(1, -1, 1, 1)
    mask = (segments.unsqueeze(0) == expanded_retained_features)
    mask = mask.sum(1) > 0
    
    # Apply mask to image
    if len(image.shape) == 3:
        mask = mask.unsqueeze(0).expand_as(image)
        masked_image = image * mask.float()
        
        # Fill removed regions
        if fill_val != 0:
            if isinstance(fill_val, (tuple, list)):
                fill_tensor = torch.tensor(fill_val, device=image.device).view(-1, 1, 1)
            else:
                fill_tensor = torch.tensor(fill_val, device=image.device)
            
            masked_image = masked_image + (1 - mask.float()) * fill_tensor
    else:
        # 2D image
        masked_image = image * mask.float()
        if fill_val != 0:
            masked_image = masked_image + (1 - mask.float()) * fill_val
    
    return masked_image


def get_patch_indices(
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int
) -> Tuple[int, int]:
    """Get the number of patches in each dimension.
    
    Args:
        image_height: Height of the image
        image_width: Width of the image
        patch_height: Height of each patch
        patch_width: Width of each patch
        
    Returns:
        Tuple of (n_patches_height, n_patches_width)
    """
    n_h = image_height // patch_height
    n_w = image_width // patch_width
    return n_h, n_w


def get_total_patches(
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int
) -> int:
    """Get total number of patches in an image.
    
    Args:
        image_height: Height of the image
        image_width: Width of the image
        patch_height: Height of each patch
        patch_width: Width of each patch
        
    Returns:
        Total number of patches
    """
    n_h, n_w = get_patch_indices(image_height, image_width, patch_height, patch_width)
    return n_h * n_w


def create_random_patch_mask(
    total_patches: int,
    removal_fraction: float,
    seed: Optional[int] = None
) -> torch.Tensor:
    """Create a random binary mask for patch removal.
    
    Args:
        total_patches: Total number of patches
        removal_fraction: Fraction of patches to remove
        seed: Random seed
        
    Returns:
        Binary mask tensor (1 = remove, 0 = keep)
    """
    if seed is not None:
        random.seed(seed)
    
    n_to_remove = int(removal_fraction * total_patches)
    n_to_remove = min(n_to_remove, total_patches - 1)  # Keep at least one patch
    
    mask = torch.zeros(total_patches, dtype=torch.int)
    remove_indices = random.sample(range(total_patches), n_to_remove)
    mask[remove_indices] = 1
    
    return mask