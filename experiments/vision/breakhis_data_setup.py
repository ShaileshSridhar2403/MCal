# Create a wrapper function to safely apply PatchCutout
from pathlib import Path
import torch
from tqdm.notebook import tqdm
import timm


# Import MCal components
from mcal.data.loaders import BreakHisLoader
from mcal.data.augmentation.patch_cutout import PatchCutout
from mcal.configs.model_dict import get_model_path
from mcal.configs.dataset_configs import get_dataset_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


from torchvision import datasets
import os
import shutil
import torchvision.transforms as transforms
from mcal.paths import DATA_ROOT

# Folder this script lives in; the BreakHis copy it trains on is under data/ here
VISION_DIR = Path(__file__).resolve().parent
# from XAI_Benchmark.datasets.dataset_utils import kaggle_setup,balance_dataframe
# from augmentation.Cutout import Cutout
# from augmentation.PatchCutout import PatchCutout


def BreakHis_full_setup(train_dir=str(VISION_DIR / "data" / "BreakHis" / "BreakHisTraining"), test_dir=str(VISION_DIR / "data" / "BreakHis" / "BreakHisTesting"), n_examples=None, train_augmentation=None, test_augmentation=None, **kwargs):
    train_dataset = None
    test_dataset = None

    if train_dir is not None:
        if not os.path.exists(train_dir):
            print(f"Dataset not found at {train_dir}, setting it up..")
            drive_path = kwargs.get("drive_path", str(DATA_ROOT / "BreakHis" / "dataset_v2.zip"))
            shutil.copy(drive_path, VISION_DIR)
            shutil.unpack_archive(VISION_DIR / "dataset_v2.zip", VISION_DIR)
            shutil.move(VISION_DIR / "dataset_v2" / "train", train_dir)
            if test_dir is not None:
                shutil.move(VISION_DIR / "dataset_v2" / "test", test_dir)
        else:
            print(f"Dataset already present at {train_dir}")

        train_transforms_list = [
            transforms.RandomRotation(7),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if train_augmentation == "PatchCutout":
            removal_fraction = kwargs.get('removal_fraction', 0.5)
            patch_size = kwargs.get('patch_size', 56)
            random_removal_fraction = kwargs.get('random_removal_fraction', False)
            random_dist = kwargs.get('random_dist', 'binomial')
            fill_val = kwargs.get('fill_val', 0)
            train_transforms_list.append(PatchCutout(patch_height=patch_size, patch_width=patch_size, removal_fraction=removal_fraction, random_removal_fraction=random_removal_fraction, random_dist=random_dist, fill_val=fill_val))
 

        train_transform = transforms.Compose(train_transforms_list)
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

    if test_dir is not None:
        if not os.path.exists(test_dir):
            print(f"Test dataset not found at {test_dir}, setting it up..")
            if not os.path.exists(VISION_DIR / "data" / "BreakHis" / "dataset_v2.zip"):
                drive_path = kwargs.get("drive_path", str(DATA_ROOT / "BreakHis" / "dataset_v2.zip"))
                shutil.copy(drive_path, VISION_DIR)
                shutil.unpack_archive(VISION_DIR / "dataset_v2.zip", VISION_DIR)
            shutil.move(VISION_DIR / "dataset_v2" / "test", test_dir)
        else:
            print(f"Test dataset already present at {test_dir}")

        test_transforms_list = [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
        if test_augmentation == "PatchCutout":
            removal_fraction = kwargs.get('removal_fraction', 0.5)
            patch_size = kwargs.get('patch_size', 56)
            random_removal_fraction = kwargs.get('random_removal_fraction', False)
            random_dist = kwargs.get('random_dist', 'binomial')
            fill_val = kwargs.get('fill_val', 0)
            test_transforms_list.append(PatchCutout(patch_height=patch_size, patch_width=patch_size, removal_fraction=removal_fraction, random_removal_fraction=random_removal_fraction, random_dist=random_dist, fill_val=fill_val))

        test_transform = transforms.Compose(test_transforms_list)
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    return train_dataset, test_dataset

def load_breakhis_data(model_type = "vanilla",fill_value=0):
    # Get dataset configuration
    breakhis_config = get_dataset_config('breakhis')
    num_classes = breakhis_config['num_classes']
    image_size = breakhis_config['image_size']

    # Load trained BreakHis model (vanilla model)
    print("📦 Loading trained BreakHis model...")
    if model_type == "vanilla":
        model_path = get_model_path('breakhis', 'vanilla')
    elif model_type == "patchcutout":
        model_path = get_model_path('breakhis', 'PatchCutout')

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

    # Initialize BreakHis data loader
    print("🧠 Loading BreakHis train dataset...")
    data_dir = DATA_ROOT
    breakhis_loader = BreakHisLoader(data_dir=data_dir)

    # Load clean test dataset (no augmentation)
    train_dataset, test_dataset_clean, _ = breakhis_loader.setup_dataset()

    print(f"✅ Dataset loaded successfully!")
    print(f"   Test samples: {len(test_dataset_clean)}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Classes: {breakhis_loader.class_names}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Image size: {image_size}")

    # Balance the dataset using the base loader method
    from torch.utils.data import Subset

    # Get all indices and labels from the dataset
    all_indices = list(range(len(train_dataset)))
    all_labels = [train_dataset[i][1] for i in all_indices]

    # Balance the dataset - set desired samples per class
    n_samples_per_class = 300  # Adjust this as needed
    balanced_indices, _ = breakhis_loader.balance_dataset(
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
    # test_loader = breakhis_loader.get_dataloader(test_dataset_clean, batch_size=batch_size, shuffle=False)
    train_loader = breakhis_loader.get_dataloader(limited_dataset, batch_size=batch_size, shuffle=True)

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Create output directory for saved images
    output_dir = Path("sample_images")
    output_dir.mkdir(exist_ok=True)
    
    print("🖼️ Loading sample BreakHis data for visualization...")
    
    # Get dataset configuration
    breakhis_config = get_dataset_config('breakhis')
    
    # Initialize BreakHis data loader
    data_dir = DATA_ROOT
    breakhis_loader = BreakHisLoader(data_dir=data_dir)
    
    # Load clean test dataset
    _, test_dataset_clean, _ = breakhis_loader.setup_dataset()
    
    # Get first sample from dataset
    sample_image, sample_label = test_dataset_clean[0]
    class_name = breakhis_loader.class_names[sample_label]

    
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
            fill_val=(0.781442, 0.633373, 0.751818) #BreakHis specific mean value
        )
        return patch_cutout(img_tensor)

 
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'BreakHis Sample - {class_name} - Ablation Fractions', fontsize=16)
    
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
        save_path = output_dir / f"breakhis_{class_name}_ablation_{fraction:.2f}.png"
        plt.imsave(save_path, img_np, cmap=cmap)
        
        print(f"Saved: {save_path}")
    
    # Save combined figure
    combined_path = output_dir / f"breakhis_{class_name}_ablation_comparison.png"
    plt.tight_layout()
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Saved ablated images to {output_dir}/")
    print(f"   Combined figure: {combined_path}")
    print(f"   Individual images: {len(ablation_fractions)} files")