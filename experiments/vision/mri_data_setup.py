# Create a wrapper function to safely apply PatchCutout
import sys
from pathlib import Path
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import timm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent

# Import MCal components
from mcal.data.loaders import MRILoader
from mcal.data.augmentation.patch_cutout import PatchCutout
from mcal.configs.model_dict import get_model_path
from mcal.configs.dataset_configs import get_dataset_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_mri_data(model_type = "vanilla",fill_value=0):
    # Get dataset configuration
    mri_config = get_dataset_config('mri')
    num_classes = mri_config['num_classes']
    image_size = mri_config['image_size']

    # Load trained MRI model (vanilla model)
    print("📦 Loading trained MRI model...")
    if model_type == "vanilla":
        model_path = get_model_path('mri', 'vanilla')
    elif model_type == "patchcutout":
        model_path = get_model_path('mri', 'PatchCutout')

    print(f"Model path: {model_path}")

    # Create Vision Transformer model
    model = timm.create_model(
        'vit_base_patch16_224', 
        pretrained=False, 
        num_classes=num_classes
    )
    



    # Load weights
    print("Loading model weights...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # Move model to device and set to eval mode
    model = model.to(device).eval()
    print("✅ Model loaded successfully!")


    def apply_patch_cutout_with_fraction(img_tensor, removal_fraction):
        """Apply PatchCutout augmentation with specific removal fraction."""
        patch_cutout = PatchCutout(
            patch_height=56,
            patch_width=56,
            removal_fraction=removal_fraction,
            random_removal_fraction=False,  # Use exact fraction, not random
            random_dist="binomial",
            fill_val=fill_value
        )
        # t
        return patch_cutout(img_tensor)


    # Generate predictions across all ablation fractions
    print("🔮 Generating predictions across all ablation fractions (0/16 to 15/16)...")

    # Initialize MRI data loader
    print("🧠 Loading MRI test dataset...")
    data_dir = project_root / "data"
    mri_loader = MRILoader(data_dir=data_dir)

    # Load clean test dataset (no augmentation)
    train_dataset, test_dataset_clean, _ = mri_loader.setup_dataset()

    print(f"✅ Dataset loaded successfully!")
    print(f"   Test samples: {len(test_dataset_clean)}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Classes: {mri_loader.class_names}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Image size: {image_size}")

    # Balance the dataset using the base loader method
    from torch.utils.data import Subset
    import numpy as np

    # Get all indices and labels from the dataset
    all_indices = list(range(len(train_dataset)))
    all_labels = [train_dataset[i][1] for i in all_indices]

    # Balance the dataset - set desired samples per class
    n_samples_per_class = 300  # Adjust this as needed
    balanced_indices, _ = mri_loader.balance_dataset(
        paths=[str(i) for i in all_indices],  # Convert indices to strings
        labels=[str(label) for label in all_labels],  # Convert to strings
        min_count=n_samples_per_class,
        randomize=True
    )

    # Convert back to integers and create subset
    balanced_indices = [int(idx) for idx in balanced_indices]
    limited_dataset = Subset(train_dataset, balanced_indices)
    
    print(f"✅ Balanced dataset created!")
    print(f"   Total balanced samples: {len(balanced_indices)}")
    print(f"   Samples per class: {n_samples_per_class}")

    # Create data loader for clean data
    batch_size = 32
    # test_loader = mri_loader.get_dataloader(test_dataset_clean, batch_size=batch_size, shuffle=False)
    train_loader = mri_loader.get_dataloader(limited_dataset, batch_size=batch_size, shuffle=True)

    # Define ablation fractions (0/16 to 15/16)
    ablation_fractions = [i/16 for i in range(16)]  # [0.0, 1/16, 2/16, ..., 15/16]
    
    all_probs = []  # Will have shape (k, n, c) where k=16, n=num_samples, c=num_classes
    true_labels = []

    # Process test set
    with torch.no_grad():
        # for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Processing batches")):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Processing batches")):
            data = data.to(device)
            target = target.to(device)
            
            batch_probs = []  # Shape: (k, batch_size, num_classes)
            
            # Generate predictions for each ablation fraction with progress bar
            for fraction in tqdm(ablation_fractions, desc=f"Batch {batch_idx+1} - Ablation levels", leave=False):
                # Create ablated data for this fraction (keep on GPU)

                ablated_data = []
                for img in data:
                    ablated_img = apply_patch_cutout_with_fraction(img, fraction)  # Keep on GPU
                    ablated_data.append(ablated_img)

                    
                ablated_data = torch.stack(ablated_data)
                
                # Get predictions for this ablation level
                output = model(ablated_data)
                prob = torch.softmax(output, dim=1)
                batch_probs.append(prob)
            
            # Stack probabilities: (k, batch_size, num_classes)
            batch_probs = torch.stack(batch_probs, dim=0)
            all_probs.append(batch_probs)
            true_labels.append(target)
            
            # Debug first batch
            if batch_idx == 0:
                print(f"Data shape: {data.shape}")
                print(f"Batch probabilities shape: {batch_probs.shape}")

    # Concatenate all predictions: (k, n, c)
    all_probs = torch.cat(all_probs, dim=1)  # Concatenate along sample dimension
    true_labels = torch.cat(true_labels, dim=0)
    print(f"✅ Generated predictions for {all_probs.shape[1]} samples with {len(ablation_fractions)} ablation levels")
    print(f"   All probabilities shape: {all_probs.shape} (k={len(ablation_fractions)}, n={all_probs.shape[1]}, c={all_probs.shape[2]})")
    print(f"   True labels shape: {true_labels.shape}")
    print(f"   Ablation fractions: {ablation_fractions}")
    return all_probs, true_labels



def load_mri_data_binomial(model_type = "vanilla",fill_value=0):
    # Get dataset configuration
    mri_config = get_dataset_config('mri')
    num_classes = mri_config['num_classes']
    image_size = mri_config['image_size']

    # Load trained MRI model (vanilla model)
    print("📦 Loading trained MRI model...")
    if model_type == "vanilla":
        model_path = get_model_path('mri', 'vanilla')
    elif model_type == "patchcutout":
        model_path = get_model_path('mri', 'PatchCutout')

    print(f"Model path: {model_path}")

    # Create Vision Transformer model
    model = timm.create_model(
        'vit_base_patch16_224', 
        pretrained=False, 
        num_classes=num_classes
    )
    



    # Load weights
    print("Loading model weights...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # Move model to device and set to eval mode
    model = model.to(device).eval()
    print("✅ Model loaded successfully!")


    def apply_patch_cutout_with_fraction(img_tensor, removal_fraction):
        """Apply PatchCutout augmentation with specific removal fraction."""
        patch_cutout = PatchCutout(
            patch_height=56,
            patch_width=56,
            removal_fraction=removal_fraction,
            random_removal_fraction=False,  # Use exact fraction, not random
            random_dist="binomial",
            fill_val=fill_value
        )
        # t
        return patch_cutout(img_tensor)


    # Generate predictions across all ablation fractions
    print("🔮 Generating predictions across all ablation fractions (0/16 to 15/16)...")

    # Initialize MRI data loader
    print("🧠 Loading MRI test dataset...")
    data_dir = project_root / "data"
    mri_loader = MRILoader(data_dir=data_dir)

    # Load clean test dataset (no augmentation)
    train_dataset, test_dataset_clean, _ = mri_loader.setup_dataset()

    print(f"✅ Dataset loaded successfully!")
    print(f"   Test samples: {len(test_dataset_clean)}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Classes: {mri_loader.class_names}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Image size: {image_size}")

    # Balance the dataset using the base loader method
    from torch.utils.data import Subset
    import numpy as np

    # Get all indices and labels from the dataset
    all_indices = list(range(len(train_dataset)))
    all_labels = [train_dataset[i][1] for i in all_indices]

    # Balance the dataset - set desired samples per class
    n_samples_per_class = 300  # Adjust this as needed
    balanced_indices, _ = mri_loader.balance_dataset(
        paths=[str(i) for i in all_indices],  # Convert indices to strings
        labels=[str(label) for label in all_labels],  # Convert to strings
        min_count=n_samples_per_class,
        randomize=True
    )

    # Convert back to integers and create subset
    balanced_indices = [int(idx) for idx in balanced_indices]
    limited_dataset = Subset(train_dataset, balanced_indices)
    
    print(f"✅ Balanced dataset created!")
    print(f"   Total balanced samples: {len(balanced_indices)}")
    print(f"   Samples per class: {n_samples_per_class}")

    # Create data loader for clean data
    batch_size = 32
    # test_loader = mri_loader.get_dataloader(test_dataset_clean, batch_size=batch_size, shuffle=False)
    train_loader = mri_loader.get_dataloader(limited_dataset, batch_size=batch_size, shuffle=True)

    # Define ablation fractions (0/16 to 15/16)
    ablation_fractions = [i/16 for i in range(16)]  # [0.0, 1/16, 2/16, ..., 15/16]
    
    all_probs = []  # Will have shape (k, n, c) where k=16, n=num_samples, c=num_classes
    true_labels = []

    # Process test set
    with torch.no_grad():
        # for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Processing batches")):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Processing batches")):
            data = data.to(device)
            target = target.to(device)
            
            batch_probs = []  # Shape: (k, batch_size, num_classes)
            
            # Generate predictions for each ablation fraction with progress bar
            for fraction in tqdm(ablation_fractions, desc=f"Batch {batch_idx+1} - Ablation levels", leave=False):
                # Create ablated data for this fraction (keep on GPU)

                fraction = np.random.binomial(16, 0.5)


                ablated_data = []
                for img in data:
                    ablated_img = apply_patch_cutout_with_fraction(img, fraction)  # Keep on GPU
                    ablated_data.append(ablated_img)

                    
                ablated_data = torch.stack(ablated_data)
                
                # Get predictions for this ablation level
                output = model(ablated_data)
                prob = torch.softmax(output, dim=1)
                batch_probs.append(prob)
            
            # Stack probabilities: (k, batch_size, num_classes)
            batch_probs = torch.stack(batch_probs, dim=0)
            all_probs.append(batch_probs)
            true_labels.append(target)
            
            # Debug first batch
            if batch_idx == 0:
                print(f"Data shape: {data.shape}")
                print(f"Batch probabilities shape: {batch_probs.shape}")

    # Concatenate all predictions: (k, n, c)
    all_probs = torch.cat(all_probs, dim=1)  # Concatenate along sample dimension
    all_probs = all_probs.reshape(-1, num_classes)
    true_labels = torch.cat(true_labels, dim=0)
    true_labels = all_probs.repeat(len(ablation_fractions))

    
    print(f"✅ Generated predictions for {all_probs.shape[1]} samples with {len(ablation_fractions)} ablation levels")
    print(f"   All probabilities shape: {all_probs.shape} (k={len(ablation_fractions)}, n={all_probs.shape[1]}, c={all_probs.shape[2]})")
    print(f"   True labels shape: {true_labels.shape}")
    print(f"   Ablation fractions: {ablation_fractions}")
    return all_probs, true_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    # Create output directory for saved images
    output_dir = Path("sample_images")
    output_dir.mkdir(exist_ok=True)
    
    print("🖼️ Loading sample MRI data for visualization...")
    
    # Get dataset configuration
    mri_config = get_dataset_config('mri')
    
    # Initialize MRI data loader
    data_dir = project_root / "data"
    mri_loader = MRILoader(data_dir=data_dir)
    
    # Load clean test dataset
    _, test_dataset_clean, _ = mri_loader.setup_dataset()
    
    # Get first sample from dataset
    sample_image, sample_label = test_dataset_clean[0]
    class_name = mri_loader.class_names[sample_label]

    
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample label: {sample_label} ({class_name})")
    
    # Convert tensor to numpy for visualization (CHW -> HWC)
    def tensor_to_numpy(tensor):
        if tensor.dim() == 3:  # CHW format
            tensor = tensor.permute(1, 2, 0)  # Convert to HWC
        return tensor.cpu().numpy()
    
    # Define ablation fractions
    ablation_fractions = [i/16 for i in range(0, 16, 2)]  # Show every 2nd fraction for clarity
    
    # Create patch cutout function
    def apply_patch_cutout_with_fraction(img_tensor, removal_fraction):
        """Apply PatchCutout augmentation with specific removal fraction."""
        patch_cutout = PatchCutout(
            patch_height=56,
            patch_width=56,
            removal_fraction=removal_fraction,
            random_removal_fraction=False,
            random_dist="binomial",
            fill_val=0 #MRI specific mean value
        )
        return patch_cutout(img_tensor)

 
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'MRI Sample - {class_name} - Ablation Fractions', fontsize=16)
    
    # Generate and save ablated images
    for idx, fraction in enumerate(ablation_fractions):
        row = idx // 4
        col = idx % 4
        
        # Apply ablation
        ablated_image = apply_patch_cutout_with_fraction(sample_image, fraction)
        
        # Convert to numpy for visualization
        img_np = tensor_to_numpy(ablated_image)
        
        # Handle grayscale vs RGB
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
            cmap = 'gray'
        else:
            cmap = None
            
        # Plot
        axes[row, col].imshow(img_np, cmap=cmap)
        axes[row, col].set_title(f'Fraction: {fraction:.2f}')
        axes[row, col].axis('off')
        
        # Save individual image
        save_path = output_dir / f"mri_{class_name}_ablation_{fraction:.2f}.png"
        plt.imsave(save_path, img_np, cmap=cmap)
        
        print(f"Saved: {save_path}")
    
    # Save combined figure
    combined_path = output_dir / f"mri_{class_name}_ablation_comparison.png"
    plt.tight_layout()
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Saved ablated images to {output_dir}/")
    print(f"   Combined figure: {combined_path}")
    print(f"   Individual images: {len(ablation_fractions)} files")



    