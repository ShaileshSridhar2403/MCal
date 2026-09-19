#!/usr/bin/env python3
"""
Simple script to ablate and save images with different fill values.
"""

from pathlib import Path
import torch
import matplotlib.pyplot as plt


from mcal.data.loaders import MRILoader
from mcal.data.augmentation.patch_cutout import PatchCutout
from mcal.paths import DATA_ROOT

def ablate_and_save():
    """Ablate single image with two different fill values and save as PNG."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data like notebook_data_loader
    data_dir = DATA_ROOT
    mri_loader = MRILoader(data_dir=data_dir)
    _, test_dataset_clean, _ = mri_loader.setup_dataset()
    
    # Get first image
    image, _ = test_dataset_clean[0]
    print(f"Image shape: {image.shape}")
    
    # Define fill values
    fill_value_black = 0
    fill_value_mean = 0.1847
    
    # Ablation fraction
    removal_fraction = 0.5
    
    # Apply ablation with black fill
    patch_cutout_black = PatchCutout(
        patch_height=56,
        patch_width=56,
        removal_fraction=removal_fraction,
        random_removal_fraction=False,
        random_dist="binomial",
        fill_val=fill_value_black
    )
    ablated_black = patch_cutout_black(image)
    
    # Apply ablation with mean fill
    patch_cutout_mean = PatchCutout(
        patch_height=56,
        patch_width=56,
        removal_fraction=removal_fraction,
        random_removal_fraction=False,
        random_dist="binomial",
        fill_val=fill_value_mean
    )
    ablated_mean = patch_cutout_mean(image)
    
    # Save original
    plt.figure(figsize=(8, 8))
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig('original.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save black fill
    plt.figure(figsize=(8, 8))
    plt.imshow(ablated_black.permute(1, 2, 0))
    plt.title('Black Fill (fill_value=0)')
    plt.axis('off')
    plt.savefig('ablated_black.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save mean fill
    plt.figure(figsize=(8, 8))
    plt.imshow(ablated_mean.permute(1, 2, 0))
    plt.title('Mean Fill (fill_value=RGB tuple)')
    plt.axis('off')
    plt.savefig('ablated_mean.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved:")
    print("- original.png")
    print("- ablated_black.png") 
    print("- ablated_mean.png")

if __name__ == "__main__":
    ablate_and_save()