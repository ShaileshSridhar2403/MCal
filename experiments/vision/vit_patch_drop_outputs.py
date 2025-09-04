import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pdb
from scipy.special import kl_div
from pathlib import Path

sys.path.append(os.path.abspath('.'))

# Add necessary paths to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).parent.parent
sys.path.append(str(project_root))

# Import required modules
from vit_patch_drop.src.models.load_trained_weights import load_vit_model, create_patch_mask
from configs.model_dict import model_dict, get_model_path
from src.data.loaders import MRILoader




def get_patch_drop_outputs(dataset_name, device, batch_size=32):
    """
    Get model outputs for a dataset with varying levels of patch dropping.
    
    Args:
        dataset: The dataset to process
        device: Device to run inference on
        batch_size: Batch size for DataLoader
        
    Returns:
        predictions, true_labels: Tensors containing model predictions and true labels

    """

    weights_path = get_model_path(dataset_name, "vanilla")
    model, device = load_vit_model(weights_path, num_classes=4, device=device)
        
    if model is None:
        print(f"Failed to load model ")
        raise ValueError("Model loading failed")
    
    # 2. Load dataset
    if dataset_name == "mri":
        data_dir = project_root / "data"
        mri_loader = MRILoader(data_dir=data_dir)

        # Load clean test dataset (no augmentation)
        _, test_dataset, _ = mri_loader.setup_dataset()
    

    else:
        # Add support for other datasets as needed
        raise ValueError(f"Dataset {dataset_name} not supported yet")
    
    print(f"Loaded test dataset with {len(test_dataset)} samples.")
    results = {}
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Total patches in a standard ViT with 224x224 image and 16x16 patches
    total_patches = 196  # 14x14 grid
    

    all_predictions = []
    all_labels = []

    # Process each fraction (0/16, 2/16, ..., 15/16)
    for fraction_num in range(0, 16):
        fraction = fraction_num / 16
        print(f"Processing with {fraction:.2f} fraction of patches dropped...")
        
        # Calculate number of patches to drop
        n_patches_to_drop = int(total_patches * fraction)
        n_patches_to_keep = total_patches - n_patches_to_drop
        
        # Store predictions and labels for this fraction
        fraction_predictions = []
        fraction_labels = []
        
        # Process batches
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Create a patch mask for this batch
            # For simplicity, we'll use the same mask for all images in the batch
            # Select random patches to keep (excluding class token)
            patches_to_keep = np.random.choice(
                range(1, total_patches + 1),  # Skip class token (index 0)
                size=n_patches_to_keep,
                replace=False
            ).tolist()
            
            # Always include class token (index 0)
            patches_to_keep = [0] + patches_to_keep
            
            # Create mask with indices to keep
            patch_mask = create_patch_mask('indices', specific_patches=patches_to_keep, 
                                          total_patches=total_patches + 1, device=device)
            
            # Run inference with the mask
            with torch.no_grad():
                outputs = model(images, patch_mask=patch_mask)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            fraction_predictions.append(probabilities)
            fraction_labels.append(labels)

        # Concatenate batch results
        predictions = torch.cat(fraction_predictions, dim=0)
        labels = torch.cat(fraction_labels, dim=0)

        all_predictions.append(predictions)
        all_labels.append(labels)


    all_probs = torch.stack(all_predictions)  # Concatenate along sample dimension
    true_labels = torch.stack(all_labels, dim=0)

    return all_probs, true_labels



if __name__ == "__main__":
    predictions, labels = get_patch_drop_outputs("mri", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), batch_size=32)
    pdb.set_trace()