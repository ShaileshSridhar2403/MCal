import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path


# Add necessary paths to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).parent.parent

# Import required modules
from vit_patch_drop.src.models.load_trained_weights import load_vit_model, create_patch_mask
from mcal.configs.model_dict import model_dict, get_model_path
from mcal.data.loaders import MRILoader, BreakHisLoader, ChexPertLoader




def get_patch_drop_outputs(dataset_name, device, batch_size=32, num_classes=None):
    """
    Get model outputs for a dataset with varying levels of patch dropping.
    
    Args:
        dataset_name: Name of the dataset ('mri', 'breakhis', 'chexpert')
        device: Device to run inference on
        batch_size: Batch size for DataLoader
        num_classes: Number of classes (auto-determined if None)
        
    Returns:
        predictions, true_labels: Tensors containing model predictions and true labels

    """
    
    # Determine number of classes based on dataset
    if num_classes is None:
        if dataset_name == "mri":
            num_classes = 4
        elif dataset_name == "breakhis":
            num_classes = 8
        elif dataset_name == "chexpert":
            num_classes = 2  # Binary classification for cardiomegaly
        else:
            raise ValueError(f"Unknown dataset {dataset_name}. Please specify num_classes.")

    weights_path = get_model_path(dataset_name, "vanilla")
    model, device = load_vit_model(weights_path, num_classes=num_classes, device=device)

    if model is None:
        print(f"Failed to load model ")
        raise ValueError("Model loading failed")
    
    # 2. Load dataset
    data_dir = project_root / "data"
    
    if dataset_name == "mri":
        mri_loader = MRILoader(data_dir=data_dir)
        # Load clean datasets (no augmentation)
        train_dataset, test_dataset, _ = mri_loader.setup_dataset()
        
        # Balance the training dataset using the same approach as other datasets
        from torch.utils.data import Subset
        import numpy as np
        
        # Get all indices and labels from the training dataset
        all_indices = list(range(len(train_dataset)))
        all_labels = [train_dataset[i][1] for i in all_indices]
        
        # Balance the dataset - set desired samples per class (same as other datasets)
        n_samples_per_class = 300
        balanced_indices, _ = mri_loader.balance_dataset(
            paths=[str(i) for i in all_indices],  # Convert indices to strings
            labels=[str(label) for label in all_labels],  # Convert to strings
            min_count=n_samples_per_class,
            randomize=True
        )
        
        # Convert back to integers and create subset
        balanced_indices = [int(idx) for idx in balanced_indices]
        test_dataset = Subset(train_dataset, balanced_indices)  # Use balanced training data
    
    elif dataset_name == "breakhis":
        breakhis_loader = BreakHisLoader(data_dir=data_dir)
        # Load clean datasets (no augmentation)
        train_dataset, test_dataset, _ = breakhis_loader.setup_dataset()
        
        # Balance the training dataset using the same approach as breakhis_data_setup
        from torch.utils.data import Subset
        import numpy as np
        
        # Get all indices and labels from the training dataset
        all_indices = list(range(len(train_dataset)))
        all_labels = [train_dataset[i][1] for i in all_indices]
        
        # Balance the dataset - set desired samples per class (same as setup file)
        n_samples_per_class = 300
        balanced_indices, _ = breakhis_loader.balance_dataset(
            paths=[str(i) for i in all_indices],  # Convert indices to strings
            labels=[str(label) for label in all_labels],  # Convert to strings
            min_count=n_samples_per_class,
            randomize=True
        )
        
        # Convert back to integers and create subset
        balanced_indices = [int(idx) for idx in balanced_indices]
        test_dataset = Subset(train_dataset, balanced_indices)  # Use balanced training data
    
    elif dataset_name == "chexpert":
        # Import and use the chexpert data setup function
        import experiments.vision.chexpert_data_setup as cds
        chexpert_loader = ChexPertLoader(data_dir=data_dir)
        # Use the chexpert full setup function to get clean datasets
        train_dataset, _ = cds.chexpert_full_setup()
        
        # Balance the training dataset using the same approach as chexpert_data_setup
        from torch.utils.data import Subset
        import numpy as np
        
        # Get all indices and labels from the training dataset
        all_indices = list(range(len(train_dataset)))
        all_labels = [train_dataset[i][1] for i in all_indices]
        
        # Balance the dataset - set desired samples per class (same as setup file)
        n_samples_per_class = 300
        balanced_indices, _ = chexpert_loader.balance_dataset(
            paths=[str(i) for i in all_indices],  # Convert indices to strings
            labels=[str(label) for label in all_labels],  # Convert to strings
            min_count=n_samples_per_class,
            randomize=True
        )
        
        # Convert back to integers and create subset
        balanced_indices = [int(idx) for idx in balanced_indices]
        test_dataset = Subset(train_dataset, balanced_indices)  # Use balanced training data

    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet. Supported datasets: mri, breakhis, chexpert")
    
    print(f"Loaded test dataset with {len(test_dataset)} samples.")
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Total patches in a standard ViT with 224x224 image and 16x16 patches
    total_patches = 196  # 14x14 grid
    

    all_predictions = []
    all_labels = []

    # Process each fraction (0/16, 2/16, ..., 15/16)
    for fraction_num in range(16):

        fraction = fraction_num / 16
        print(f"Processing with {fraction} fraction of patches dropped...")
        
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


    for fraction_idx in range(predictions.shape[0]):
        fraction_preds = predictions[fraction_idx]
        fraction_labels = labels[fraction_idx]

        predicted_labels = torch.argmax(fraction_preds, dim=-1)
        accuracy = (predicted_labels == fraction_labels).float().mean().item()
        print(f"Fraction {fraction_idx}/16 - Accuracy: {accuracy*100:.2f}%")




