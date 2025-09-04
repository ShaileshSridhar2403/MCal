#!/usr/bin/env python3
"""
Calculate mean pixel values for MRI dataset.

This script loads the entire MRI training dataset and calculates
the mean pixel value across all images for each RGB channel.
"""

import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))
sys.path.insert(0, str(mcal_root / "src"))
import pdb

# Import MCal data loaders
from src.data.loaders import MRILoader

def calculate_mri_mean_pixel_values():
    """Calculate mean pixel values for the MRI training dataset.
    
    Returns:
        tuple: (R, G, B) mean pixel values across the entire training dataset
    """
    
    # Load MRI training dataset
    print("Loading MRI training dataset...")
    data_dir = mcal_root / "data"
    mri_loader = MRILoader(data_dir=data_dir)
    
    # Load training dataset
    train_dataset, _, _ = mri_loader.setup_dataset()
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Initialize running sum
    running_sum = torch.zeros(3)
    
    print("Calculating mean pixel values...")
    for i, (image, _) in enumerate(tqdm(train_dataset, desc="Processing images")):
        # We KNOW that image has shape (3,224,224)
        # pdb.set_trace()
        image = image.reshape(3, -1)  # Changes image to shape (3, 224*224)
        running_sum += image.mean(dim=1)  # Shape (3,)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(train_dataset)} images")
    
    # Calculate average
    three_channel_avg = running_sum / len(train_dataset)  # Shape (3,)
    
    # Extract RGB values
    r_mean, g_mean, b_mean = three_channel_avg.tolist()
    
    print(f"\nMean pixel values calculated from {len(train_dataset)} images")
    print(f"R channel mean: {r_mean:.6f}")
    print(f"G channel mean: {g_mean:.6f}")  
    print(f"B channel mean: {b_mean:.6f}")
    print(f"RGB tuple: ({r_mean:.6f}, {g_mean:.6f}, {b_mean:.6f})")
    
    return (r_mean, g_mean, b_mean)


def main():
    """Main execution function."""
    try:
        # Calculate mean pixel values
        rgb_means = calculate_mri_mean_pixel_values()
        print(f"\nFinal result: {rgb_means}")
        return 0
        
    except Exception as e:
        print(f"Error calculating mean pixel values: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())