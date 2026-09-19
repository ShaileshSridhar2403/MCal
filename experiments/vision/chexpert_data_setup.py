from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import shutil
import subprocess
import torchvision.transforms as transforms
# from augmentation.PatchCutout import PatchCutout

import timm
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import sys
from pathlib import Path

#potential issues:
#data is not balanced
#we were previously using train data for table results and now using test data

project_root = Path().absolute().parent.parent


from mcal.configs.dataset_configs import get_dataset_config
from mcal.data.augmentation.patch_cutout import PatchCutout
from mcal.data.loaders import ChexPertLoader
from mcal.configs.model_dict import get_model_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def balance_dataframe(df,task,n = None):
    if n is None:
        min_row_count = min(df[task].value_counts())
    else:
        min_row_count = n

    df = (df.groupby(task, as_index=False)
        .apply(lambda x: x.sample(min_row_count))
        .reset_index(drop=True))
    
    return df


def download_and_unpack_chexpert_dataset():
    #takes ~5 min
    logs = subprocess.run("kaggle datasets download -d willarevalo/chexpert-v10-small".split())
    print(logs)
    shutil.unpack_archive("chexpert-v10-small.zip", ".")
    print("Downloaded and Unpacked dataset")



class ChestXrayDataset(Dataset):

    def __init__(self, folder_dir, dataframe, image_size, normalization,task="Cardiomegaly",multilabel = False,max_n = None,transform=None):
        """
        Init Dataset

        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """

        self.image_paths = [] # List of image paths
        self.image_labels = [] # List of image labels
        self.transform = transform

        # Get all image paths and image labels from dataframe
        dataframe = dataframe.loc[dataframe[task].isin([0,1])]
       
        df = balance_dataframe(dataframe,task,max_n)
        self.image_paths = df.Path.apply(lambda path : os.path.join(folder_dir, path)).tolist()
        self.image_labels = [ i for i in df[task].apply(int).tolist()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """

        # Read image
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB") # Convert image to RGB channels

        # TODO: Image augmentation code would be placed here

        # Resize and convert image to torch tensor
        image_data = self.transform(image_data)

        return image_data, self.image_labels[index]


def chexpert_full_setup(train_dir='./CheXpert-v1.0-small/train', test_dir="./CheXpert-v1.0-small/valid", n_examples=None, train_augmentation=None, test_augmentation=None, **kwargs):
    train_dataset = None
    test_dataset = None

    if train_dir is not None:
        if not os.path.isdir(train_dir):
            print(f"No dataset found at {train_dir}, proceeding to download")
            download_and_unpack_chexpert_dataset()
        else:
            print("Existing Downloaded chexpert dataset found, proceeding with Data Processing")

        train_transforms_list = [
            transforms.RandomRotation(7),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5013, 0.5013, 0.5013], std=[0.2908, 0.2908, 0.2908])
        ]

        train_data = pd.read_csv("./CheXpert-v1.0-small/train.csv")
        train_dataset = ChestXrayDataset(".", train_data, 224, True, max_n=3000,
                                         transform=transforms.Compose(train_transforms_list))

    if test_dir is not None:
        if not os.path.isdir(test_dir):
            print(f"No dataset found at {test_dir}, proceeding to download")
            download_and_unpack_chexpert_dataset()
        else:
            print("Existing Downloaded chexpert test dataset found, proceeding with Data Processing")

        test_transforms_list = [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5013, 0.5013, 0.5013], std=[0.2908, 0.2908, 0.2908])
        ]


        if test_augmentation == "PatchCutout":
            removal_fraction = kwargs.get('removal_fraction', 0)
            patch_size = kwargs.get('patch_size', 56)
            random_removal_fraction = kwargs.get('random_removal_fraction', False)
            random_dist = kwargs.get('random_dist', 'binomial')
            test_transforms_list.insert(-1, PatchCutout(patch_height=patch_size, patch_width=patch_size, removal_fraction=removal_fraction, random_removal_fraction=random_removal_fraction, random_dist=random_dist, fill_val=fill_value))

        val_data = pd.read_csv("./CheXpert-v1.0-small/valid.csv")
        test_dataset = ChestXrayDataset(".", val_data, 224, True,
                                        transform=transforms.Compose(test_transforms_list))

    return train_dataset, test_dataset





def load_chexpert_data(model_type='vanilla', fill_value=0):
    '''
    Loads Chexpert data, ablated wrt all fractions from 0/16 to 15/16
    '''
    train_dataset, test_dataset = chexpert_full_setup(fill_value=fill_value, model_type=model_type)
    # ChexPertLoader


    chexpert_config = get_dataset_config('chexpert')
    num_classes = chexpert_config['num_classes']
    image_size = chexpert_config['image_size']

    # Load trained chexpert model (vanilla model)
    print("📦 Loading trained chexpert model...")
    if model_type == "vanilla":
        model_path = get_model_path('chexpert', 'vanilla')
    elif model_type == "patchcutout":
        model_path = get_model_path('chexpert', 'PatchCutout')

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
        return patch_cutout(img_tensor)


    # Generate predictions across all ablation fractions
    print("🔮 Generating predictions across all ablation fractions (0/16 to 15/16)...")

    # Initialize chexpert data loader
    print("🧠 Loading chexpert test dataset...")
    data_dir = project_root / "data"
    chexpert_loader = ChexPertLoader(data_dir=data_dir)

    # Load clean test dataset (no augmentation)
    # _, test_dataset_clean, _ = chexpert_loader.setup_dataset()
    train_dataset, test_dataset_clean= chexpert_full_setup()

    print(f"✅ Dataset loaded successfully!")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset_clean)}")
    print(f"   Classes: {chexpert_loader.class_names}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Image size: {image_size}")

    # Balance the dataset using ChexPert's existing balanced approach
    from torch.utils.data import Subset
    import numpy as np

    # Get all indices and labels from the dataset
    all_indices = list(range(len(train_dataset)))
    all_labels = [train_dataset[i][1] for i in all_indices]

    # Balance the dataset - set desired samples per class  
    n_samples_per_class = 300  # Adjust this as needed
    balanced_indices, _ = chexpert_loader.balance_dataset(
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
    # test_loader = chexpert_loader.get_dataloader(test_dataset_clean, batch_size=batch_size, shuffle=False)
    train_loader = chexpert_loader.get_dataloader(limited_dataset, batch_size=batch_size, shuffle=True)

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
                    ablated_img = transforms.Normalize(mean=[0.5013, 0.5013, 0.5013], std=[0.2908, 0.2908, 0.2908])(ablated_img)
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
    import numpy as np
    from pathlib import Path
    
    # Create output directory for saved images
    output_dir = Path("sample_images")
    output_dir.mkdir(exist_ok=True)
    
    print("🖼️ Loading sample chexpert data for visualization...")
    
    # Get dataset configuration
    chexpert_config = get_dataset_config('chexpert')
    
    # Initialize chexpert data loader
    data_dir = project_root / "data"
    chexpert_loader = ChexPertLoader(data_dir=data_dir)

    # Load clean test dataset
    # _, test_dataset_clean, _ = chexpert_loader.setup_dataset()
    _, test_dataset_clean= chexpert_full_setup()
    
    # Get first sample from dataset
    sample_image, sample_label = test_dataset_clean[0]
    class_name = chexpert_loader.class_names[sample_label]


    
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample label: {sample_label} ({class_name})")
    #
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
            fill_val=0 #chexpert specific mean value
        )
        return patch_cutout(img_tensor)

    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Sample - {class_name} - Ablation Fractions', fontsize=16)
    
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
        save_path = output_dir / f"chexpert_{class_name}_ablation_{fraction:.2f}.png"
        plt.imsave(save_path, img_np, cmap=cmap)
        
        print(f"Saved: {save_path}")
    
    # Save combined figure
    combined_path = output_dir / f"chexpert_{class_name}_ablation_comparison.png"
    plt.tight_layout()
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Saved ablated images to {output_dir}/")
    print(f"   Combined figure: {combined_path}")
    print(f"   Individual images: {len(ablation_fractions)} files")