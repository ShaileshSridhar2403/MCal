#!/usr/bin/env python3
"""
Unified Data Loader for MCal Experiments

A single file containing all data loading functions with a consistent API.
Just loads raw data - no models, no predictions, just data.
"""

from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import random
from tqdm import tqdm
import pandas as pd
from PIL import Image
import datasets as huggingface_datasets
from transformers import AutoTokenizer


# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent

from mcal.data.augmentation.patch_cutout import PatchCutout

# Default data directory
DATA_ROOT = PROJECT_ROOT / "data"


# =============================================================================
# MRI DATASET
# =============================================================================


def _mask_random_patches_prob(image, mask_prob=0.5, patch_size=16, fill_val=0, seed=None):
    """Alternative implementation using F.interpolate for patch masking."""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    C, H, W = image.shape
    # assert H % patch_size == 0 and W % patch_size == 0, f"Image dimensions must be multiples of patch_size {patch_size}"
    n_patches_h, n_patches_w = H // patch_size, W // patch_size
    patch_mask = torch.rand(n_patches_h, n_patches_w) < mask_prob
    mask_full = F.interpolate(
        patch_mask.float().view(1, 1, n_patches_h, n_patches_w),
        size=(H, W),
        mode='nearest'
    ).view(H, W)

    masked_image = image.clone()
    masked_image[:, mask_full.bool()] = fill_val
    return masked_image


def _mask_random_patches_exact(image, mask_prob=0.5, patch_size=16, fill_val=0, seed=None):
    """Alternative implementation using F.interpolate for patch masking."""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    C, H, W = image.shape
    # assert H % patch_size == 0 and W % patch_size == 0, f"Image dimensions must be multiples of patch_size {patch_size}"

    n_patches_h, n_patches_w = H // patch_size, W // patch_size
    num_to_replace = max(0, int(n_patches_h * n_patches_w * mask_prob))
    patch_mask = torch.zeros(n_patches_h * n_patches_w)

    if num_to_replace > 0:
        replace_indices = random.sample(range(n_patches_h * n_patches_w), num_to_replace)
        patch_mask[replace_indices] = 1

    mask_full = F.interpolate(
        patch_mask.view(1, 1, n_patches_h, n_patches_w),
        size=(H, W),
        mode='nearest'
    ).view(H, W)

    masked_image = image.clone()
    masked_image[:, mask_full.bool()] = fill_val
    return masked_image


class MRICleanDataset(Dataset):
    def __init__(self, split='test', n_samples=None):
        # Map split names to actual directory names
        if split.lower() == 'train' or split == 'Training':
            data_dir_name = 'Training'
        elif split.lower() == 'test' or split == 'Testing':
            data_dir_name = 'Testing'
        else:
            data_dir_name = split

        self.data_dir = str(Path(__file__).parent / "vision" / "data" / data_dir_name)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.ImageFolder(self.data_dir, transform=self.transforms)
        # Cap n_samples at actual dataset size
        self.n_samples = min(n_samples, len(self.dataset)) if n_samples is not None else len(self.dataset)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.dataset[idx]


class MRIPatchedProbDataset(Dataset):
    def __init__(self, split='test', n_samples=None, p_ablate=0.5, patch_size=56, fill_val=0, seed=None):
        self.dataset = MRICleanDataset(split, n_samples)
        self.p_ablate = p_ablate
        self.patch_size = patch_size
        self.fill_val = fill_val
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_masked = _mask_random_patches_prob(
            image,
            self.p_ablate,
            self.patch_size,
            self.fill_val,
            self.seed + idx if self.seed is not None else None
        )
        return image_masked, label


class MRIPatchedExactDataset(Dataset):
    def __init__(self, split='test', n_samples=None, p_ablate=0.5, patch_size=56, fill_val=0, seed=None):
        self.dataset = MRICleanDataset(split, n_samples)
        self.p_ablate = p_ablate
        self.patch_size = patch_size
        self.fill_val = fill_val
        self.seed = seed
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_masked = _mask_random_patches_exact(
            image,
            self.p_ablate,
            self.patch_size,
            self.fill_val,
            self.seed + idx if self.seed is not None else None
        )
        return image_masked, label


def load_mri_clean(split='test', n_samples=None):
    """
    Load clean MRI dataset without any augmentation.

    Args:
        split: 'train' or 'test'
        n_samples: Number of samples to load (None = all)
    """
    return MRICleanDataset(split, n_samples)


def load_mri_ablated_prob(split='test', p_ablate=0.5, n_samples=None, patch_size=56, fill_val=0, seed=None):
    """
    Load MRI with probabilistic patch ablation.

    Args:
        split: 'train' or 'test'
        p_ablate: Probability of ablating each patch (0.0 to 1.0)
        n_samples: Number of samples to load
        patch_size: Size of patches to ablate
        fill_val: Value to fill masked patches with
        seed: Random seed for reproducibility

    Returns:
        dataset: PyTorch dataset with patch ablation applied
    """
    # Load the base dataset
    base_dataset = load_mri_clean(split, n_samples)
    
    # Create a custom dataset class that applies patch masking
    class PatchedDataset:
        def __init__(self, base_dataset, mask_prob, patch_size, fill_val, seed):
            self.base_dataset = base_dataset
            self.mask_prob = mask_prob
            self.patch_size = patch_size
            self.fill_val = fill_val
            self.seed = seed
            
        def __len__(self):
            return len(self.base_dataset)
            
        def __getitem__(self, idx):
            image, label = self.base_dataset[idx]
            # Apply patch masking with the given seed offset by index for variety
            masked_image = _mask_random_patches_prob(
                image, 
                mask_prob=self.mask_prob, 
                patch_size=self.patch_size, 
                fill_val=self.fill_val, 
                seed=self.seed + idx if self.seed is not None else None
            )
            return masked_image, label
    
    return PatchedDataset(base_dataset, p_ablate, patch_size, fill_val, seed)


def load_mri_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None, patch_size=56, fill_val=0, seed=None):
    """
    Load MRI with exact fraction of patches ablated.

    Args:
        split: 'train' or 'test'
        fraction_ablate: Exact fraction of patches to ablate (0.0 to 1.0)
        n_samples: Number of samples to load
        patch_size: Size of patches to ablate
        fill_val: Value to fill masked patches with
        seed: Random seed for reproducibility

    Returns:
        dataset: PyTorch dataset with exact patch ablation applied
    """
    # Load the base dataset
    base_dataset = load_mri_clean(split, n_samples)
    
    # Create a custom dataset class that applies exact patch masking
    class ExactPatchedDataset:
        def __init__(self, base_dataset, mask_prob, patch_size, fill_val, seed):
            self.base_dataset = base_dataset
            self.mask_prob = mask_prob
            self.patch_size = patch_size
            self.fill_val = fill_val
            self.seed = seed
            
        def __len__(self):
            return len(self.base_dataset)
            
        def __getitem__(self, idx):
            image, label = self.base_dataset[idx]
            # Apply exact patch masking with the given seed offset by index for variety
            masked_image = _mask_random_patches_exact(
                image, 
                mask_prob=self.mask_prob, 
                patch_size=self.patch_size, 
                fill_val=self.fill_val, 
                seed=self.seed + idx if self.seed is not None else None
            )
            return masked_image, label
    
    return ExactPatchedDataset(base_dataset, fraction_ablate, patch_size, fill_val, seed)


def load_mri_fractionwise(split='test', n_fractions=16, n_samples=None):
    """
    Load MRI with multiple ablation fractions (for KL experiments).

    Args:
        split: 'train' or 'test'
        n_fractions: Number of ablation levels (0/n to (n-1)/n)
        n_samples: Number of samples to load

    Returns:
        images: (n_fractions, n_samples, 3, 224, 224) tensor
        labels: (n_samples,) tensor
    """
    clean_images, labels = load_mri_clean(split, n_samples)

    n_samples = len(labels)
    ablated_images = torch.zeros(n_fractions, n_samples, 3, 224, 224)

    for i in range(n_fractions):
        fraction = i / n_fractions
        if fraction == 0:
            ablated_images[i] = clean_images
        else:
            augmenter = PatchCutout(
                patch_height=56,
                patch_width=56,
                removal_fraction=fraction,
                random_removal_fraction=False,
                fill_val=0
            )
            ablated_images[i] = torch.stack([augmenter(img) for img in clean_images])

    return ablated_images, labels


def load_mri_clean_and_ablated(split='test', p_ablate=0.5, n_samples=None, patch_size=16, fill_val=0, seed=None, shuffle=False):
    """
    Load clean MRI images and their ablated versions with probabilistic patch masking.

    Ensures clean and ablated images are perfectly aligned (same samples).

    Args:
        split: 'train' or 'test'
        p_ablate: Probability of masking each patch (default 0.5)
        n_samples: Number of samples to load
        patch_size: Size of patches to mask (default 16)
        fill_val: Value to fill masked patches with
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the data before returning (default False)

    Returns:
        images_clean: (n_samples, 3, 224, 224) tensor of clean images
        images_ablated: (n_samples, 3, 224, 224) tensor of ablated images
        labels: (n_samples,) tensor of labels
    """
    # Load clean dataset
    dataset = load_mri_clean(split, n_samples)

    # Convert dataset to tensors
    images_list = []
    labels_list = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        images_list.append(img)
        labels_list.append(label)

    images_clean = torch.stack(images_list)
    labels = torch.tensor(labels_list)

    # Shuffle if requested
    if shuffle:
        if seed is not None:
            torch.manual_seed(seed)
        perm = torch.randperm(len(images_clean))
        images_clean = images_clean[perm]
        labels = labels[perm]

    # Create ablated versions with probabilistic patch masking
    images_ablated = images_clean.clone()
    C, H, W = images_clean.shape[1:]
    n_patches_h, n_patches_w = H // patch_size, W // patch_size

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    for i in range(len(images_ablated)):
        # Generate random mask for patches
        patch_mask = torch.rand(n_patches_h, n_patches_w) < p_ablate
        patch_mask = patch_mask.unsqueeze(0).repeat(C, 1, 1)
        # Upsample to full image size
        patch_mask = F.interpolate(
            patch_mask.unsqueeze(0).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze(0).bool()
        # Apply mask
        images_ablated[i, patch_mask] = fill_val

    return images_clean, images_ablated, labels


# =============================================================================
# CHEXPERT DATASET
# =============================================================================

def load_chexpert_clean(split='test', n_samples=None):
    """
    Load clean CheXpert dataset without any augmentation.

    Note: Currently only loads validation set for Cardiomegaly classification.
    """
    # CheXpert data location
    chexpert_dir = Path(__file__).parent / "vision" / "CheXpert-v1.0-small"

    if split == 'train':
        csv_file = chexpert_dir / "train.csv"
        # Limit train samples to avoid memory issues
        max_samples = min(n_samples, 3000) if n_samples else 3000
    else:
        csv_file = chexpert_dir / "valid.csv"
        max_samples = n_samples

    # Read CSV and filter for Cardiomegaly task (0 or 1, excluding uncertain)
    df = pd.read_csv(csv_file)
    df = df[df['Cardiomegaly'].isin([0.0, 1.0])]

    # Take first max_samples deterministically
    if max_samples:
        df = df.head(min(max_samples, len(df)))

    # Load images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    images = []
    labels = []

    for idx, row in df.iterrows():
        # Path is relative to CheXpert-v1.0-small directory
        img_path = chexpert_dir.parent / row['Path']

        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(int(row['Cardiomegaly']))

    if len(images) == 0:
        raise ValueError(f"No CheXpert images found. Please check data at {chexpert_dir}")

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def load_chexpert_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load CheXpert with probabilistic patch ablation.
    """
    images, labels = load_chexpert_clean(split, n_samples)

    augmenter = PatchCutout(
        patch_height=56,
        patch_width=56,
        removal_fraction=p_ablate,
        random_removal_fraction=False,
        fill_val=0
    )

    ablated_images = torch.stack([augmenter(img) for img in images])
    return ablated_images, labels


def load_chexpert_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load CheXpert with exact fraction of patches ablated.
    """
    return load_chexpert_ablated_prob(split, fraction_ablate, n_samples)


def load_chexpert_fractionwise(split='test', n_fractions=16, n_samples=None):
    """
    Load CheXpert with multiple ablation fractions.
    """
    clean_images, labels = load_chexpert_clean(split, n_samples)

    n_samples = len(labels)
    ablated_images = torch.zeros(n_fractions, n_samples, 3, 224, 224)

    for i in range(n_fractions):
        fraction = i / n_fractions
        if fraction == 0:
            ablated_images[i] = clean_images
        else:
            augmenter = PatchCutout(
                patch_height=56,
                patch_width=56,
                removal_fraction=fraction,
                random_removal_fraction=False,
                fill_val=0
            )
            ablated_images[i] = torch.stack([augmenter(img) for img in clean_images])

    return ablated_images, labels


# =============================================================================
# BREAKHIS DATASET
# =============================================================================

def load_breakhis_clean(split='test', n_samples=None):
    """
    Load clean BreakHis dataset without any augmentation.
    """
    # BreakHis data is in dataset_v2/ at project root
    if split == 'train':
        data_dir = PROJECT_ROOT / "dataset_v2" / "train"
    else:
        data_dir = PROJECT_ROOT / "dataset_v2" / "test"

    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)

    if n_samples:
        # Use deterministic subset - take first n_samples
        indices = list(range(min(n_samples, len(dataset))))
        dataset = Subset(dataset, indices)

    return _dataset_to_tensors(dataset)


def load_breakhis_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load BreakHis with probabilistic patch ablation.
    """
    images, labels = load_breakhis_clean(split, n_samples)

    augmenter = PatchCutout(
        patch_height=56,
        patch_width=56,
        removal_fraction=p_ablate,
        random_removal_fraction=False,
        fill_val=0
    )

    ablated_images = torch.stack([augmenter(img) for img in images])
    return ablated_images, labels


def load_breakhis_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load BreakHis with exact fraction of patches ablated.
    """
    return load_breakhis_ablated_prob(split, fraction_ablate, n_samples)


def load_breakhis_fractionwise(split='test', n_fractions=16, n_samples=None):
    """
    Load BreakHis with multiple ablation fractions.
    """
    clean_images, labels = load_breakhis_clean(split, n_samples)

    n_samples = len(labels)
    ablated_images = torch.zeros(n_fractions, n_samples, 3, 224, 224)

    for i in range(n_fractions):
        fraction = i / n_fractions
        if fraction == 0:
            ablated_images[i] = clean_images
        else:
            augmenter = PatchCutout(
                patch_height=56,
                patch_width=56,
                removal_fraction=fraction,
                random_removal_fraction=False,
                fill_val=0
            )
            ablated_images[i] = torch.stack([augmenter(img) for img in clean_images])

    return ablated_images, labels


# =============================================================================
# MEDQA DATASET
# =============================================================================

def load_medqa_clean(split='test', n_samples=None):
    """
    Load clean MedQA dataset without any text manipulation.
    Loads from HuggingFace: bigbio/med_qa (5-option format).
    """
    # Map split names for HuggingFace
    if split == 'dev' or split == 'test':
        hf_split = 'validation'
    elif split == 'train':
        hf_split = 'train'
    else:
        hf_split = split

    # Load from HuggingFace
    dataset = huggingface_datasets.load_dataset(
        'bigbio/med_qa',
        'med_qa_en_source',  # English 5-option version
        split=hf_split,
        trust_remote_code=True
    )

    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    return dataset


def load_medqa_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load MedQA with probabilistic token removal.
    """
    dataset = load_medqa_clean(split, n_samples)
    return dataset.map(lambda x: {
        'question': [_mask_random_words_prob(
            q,
            replacement_token='UNKWORDZ',
            mask_prob=p_ablate,
            seed=42
        ) for q in x['question']],
    }, batched=True)



def load_medqa_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load MedQA with exact fraction of tokens removed.
    """
    dataset = load_medqa_clean(split, n_samples)
    return dataset.map(lambda x: {
        'question': [_mask_random_words_exact(
            q,
            replacement_token='UNKWORDZ',
            mask_prob=fraction_ablate,
            seed=42
        ) for q in x['question']],
    }, batched=True)


def load_medqa_fractionwise(split='test', n_fractions=16, n_samples=None):
    """
    Load MedQA with multiple ablation fractions.
    """
    dataset = load_medqa_clean(split, n_samples)

    modified_datasets = [
        dataset.map(lambda x: {
            'question': [_mask_random_words_exact(
                q,
                replacement_token='UNKWORDZ',
                mask_prob=i / n_fractions,
                seed=42
            ) for q in x['question']],
        }, batched=True)
        for i in range(n_fractions)
    ]

    return modified_datasets


def load_medqa_clean_and_ablated(split='test', p_ablate=0.5, n_samples=None, seed=None, shuffle=False):
    """
    Load clean MedQA and ablated versions with probabilistic token masking.

    Ensures clean and ablated datasets are perfectly aligned (same samples).

    Args:
        split: 'test' or 'train'
        p_ablate: Probability of masking each token (default 0.5)
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the data before returning (default False)

    Returns:
        dataset_clean: HuggingFace dataset with clean questions
        dataset_ablated: HuggingFace dataset with ablated questions (same order)
    """
    # Load clean dataset
    dataset_clean = load_medqa_clean(split, n_samples)

    # Shuffle if requested
    if shuffle:
        if seed is not None:
            dataset_clean = dataset_clean.shuffle(seed=seed)
        else:
            dataset_clean = dataset_clean.shuffle()

    # Create ablated version with probabilistic word masking
    if seed is not None:
        random.seed(seed)

    def ablate_question(example, idx):
        words = example['question'].split()
        n_keep = int(len(words) * (1 - p_ablate))
        if n_keep > 0 and len(words) > 0:
            # Use idx as part of seed for reproducibility
            rng = random.Random(seed + idx if seed is not None else None)
            keep_indices = set(rng.sample(range(len(words)), n_keep))
            ablated_words = [words[i] if i in keep_indices else '' for i in range(len(words))]
            ablated_q = ' '.join([w for w in ablated_words if w])
        else:
            ablated_q = ''
        return {'question': ablated_q}

    dataset_ablated = dataset_clean.map(ablate_question, with_indices=True)

    return dataset_clean, dataset_ablated


# =============================================================================
# MEDMCQA DATASET
# =============================================================================

def load_medmcqa_clean(split='test', n_samples=None):
    """
    Load clean MedMCQA dataset without any text manipulation.
    Returns HuggingFace dataset.
    """
    # Map split names
    if split == 'test':
        hf_split = 'validation'
    elif split == 'train':
        hf_split = 'train'
    else:
        hf_split = 'validation'

    # Load from HuggingFace datasets
    dataset = huggingface_datasets.load_dataset("openlifescienceai/medmcqa", split=hf_split, trust_remote_code=True)
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    return dataset

def load_medmcqa_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load MedMCQA with probabilistic token removal.
    """
    dataset = load_medmcqa_clean(split, n_samples)
    return dataset.map(lambda x: {
        'question': [_mask_random_words_prob(
            q,
            replacement_token='UNKWORDZ',
            mask_prob=p_ablate,
            seed=42
        ) for q in x['question']],
    }, batched=True)
    

def load_medmcqa_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load MedMCQA with exact fraction of tokens removed.
    """
    dataset = load_medmcqa_clean(split, n_samples)
    return dataset.map(lambda x: {
        'question': [_mask_random_words_exact(
            q,
            replacement_token='UNKWORDZ',
            mask_prob=fraction_ablate,
            seed=42
        ) for q in x['question']],
    }, batched=True)


def load_medmcqa_fractionwise(split='test', n_fractions=16, n_samples=None):
    """
    Load MedMCQA with multiple ablation fractions.
    """
    dataset = load_medmcqa_clean(split, n_samples)

    modified_datasets = [
        dataset.map(lambda x: {
            'question': [_mask_random_words_exact(
                q,
                replacement_token='UNKWORDZ',
                mask_prob=i / n_fractions,
                seed=42
            ) for q in x['question']],
        }, batched=True)
        for i in range(n_fractions)
    ]

    return modified_datasets


# =============================================================================
# CTG (CARDIOTOCOGRAPHY) DATASET
# =============================================================================

def load_ctg_clean(split='test', n_samples=None):
    """
    Load clean CTG (Cardiotocography) dataset without any missingness.

    Args:
        split: 'train' or 'test' - splits data 80/20
        n_samples: Number of samples to load from the split

    Returns:
        X: Features as numpy array (n_samples, n_features)
        y: Labels as numpy array (n_samples,)
    """
    # Load CTG features and targets
    features_file = PROJECT_ROOT / 'experiments' / 'data' / 'ctg_features.csv'
    targets_file = PROJECT_ROOT / 'experiments' / 'data' / 'ctg_targets.csv'

    if not features_file.exists() or not targets_file.exists():
        raise FileNotFoundError(f"CTG data files not found at {features_file.parent}")

    # Load data
    features_df = pd.read_csv(features_file)
    targets_df = pd.read_csv(targets_file)

    # Combine features and targets
    X = features_df.values
    y = targets_df['NSP'].values - 1  # Convert to 0-indexed (1,2,3 -> 0,1,2)

    # Split train/test (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Select split
    if split == 'train':
        X_split, y_split = X_train, y_train
    else:
        X_split, y_split = X_test, y_test

    # Sample if needed
    if n_samples is not None and n_samples < len(X_split):
        indices = np.random.RandomState(42).choice(len(X_split), n_samples, replace=False)
        X_split = X_split[indices]
        y_split = y_split[indices]

    return X_split, y_split


def load_ctg_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load CTG with probabilistic feature missingness (binomial).

    Args:
        split: 'train' or 'test'
        p_ablate: Probability of ablating each feature (0.0 to 1.0)
        n_samples: Number of samples to load

    Returns:
        X: Features with probabilistic missingness (n_samples, n_features)
        y: Labels as numpy array (n_samples,)
    """
    X_clean, y = load_ctg_clean(split, n_samples)

    # Apply probabilistic missingness
    X_ablated = X_clean.copy()
    n_samples_actual, n_features = X_ablated.shape

    for i in range(n_samples_actual):
        # Each feature has p_ablate probability of being masked (binomial)
        mask = np.random.random(n_features) < p_ablate
        X_ablated[i, mask] = 0

    return X_ablated, y


def load_ctg_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load CTG with exact fraction of features ablated.

    Args:
        split: 'train' or 'test'
        fraction_ablate: Exact fraction of features to ablate (0.0 to 1.0)
        n_samples: Number of samples to load

    Returns:
        X: Features with exact missingness (n_samples, n_features)
        y: Labels as numpy array (n_samples,)
    """
    X_clean, y = load_ctg_clean(split, n_samples)

    # Apply exact missingness
    X_ablated = X_clean.copy()
    n_samples_actual, n_features = X_ablated.shape
    n_to_ablate = int(n_features * fraction_ablate)

    for i in range(n_samples_actual):
        if n_to_ablate > 0:
            # Randomly select exact number of features to mask
            ablate_indices = np.random.choice(n_features, n_to_ablate, replace=False)
            X_ablated[i, ablate_indices] = 0

    return X_ablated, y


def load_ctg_clean_and_ablated(split='test', p_ablate=0.5, n_samples=None, seed=None, shuffle=False):
    """
    Load clean CTG data and ablated versions with probabilistic feature masking.

    Ensures clean and ablated data are perfectly aligned (same samples).

    Args:
        split: 'train' or 'test'
        p_ablate: Probability of masking each feature (default 0.5)
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the data before returning (default False)

    Returns:
        X_clean: Clean features (n_samples, n_features)
        X_ablated: Ablated features (n_samples, n_features)
        y: Labels as numpy array (n_samples,)
    """
    # Load clean data
    X_clean, y = load_ctg_clean(split, n_samples)

    # Shuffle if requested
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X_clean))
        X_clean = X_clean[perm]
        y = y[perm]

    # Create ablated version with probabilistic feature masking
    X_ablated = X_clean.copy()
    n_samples_actual, n_features = X_ablated.shape

    if seed is not None:
        np.random.seed(seed)

    for i in range(n_samples_actual):
        # Probabilistic masking - each feature independently masked with prob p_ablate
        mask = np.random.random(n_features) < p_ablate
        X_ablated[i, mask] = 0

    return X_clean, X_ablated, y


# =============================================================================
# PHYSIONET DATASET
# =============================================================================

def _load_physionet_from_missingness_files(missingness_dir):
    """
    Load PhysioNet data from missingness level files (0-30% missingness range).
    Internal helper function - loads base data source.
    """
    import glob

    # List all relevant CSV files
    file_pattern = str(Path(missingness_dir) / "missingness_0*.csv.gz")
    files = glob.glob(file_pattern)

    # Filter files for 0-30% missingness
    files = [f for f in files if int(Path(f).stem.split('_')[1][:3]) <= 30]

    if not files:
        raise ValueError(f"No PhysioNet files found in {missingness_dir} for 0-30% missingness range")

    # Load and concatenate dataframes
    dfs = []
    for file in files:
        df = pd.read_csv(file, compression='gzip', index_col=0)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)
    return combined_df


def _balance_physionet_dataset(df, target_column='In-hospital_death'):
    """
    Balance PhysioNet dataset by undersampling majority class.
    Internal helper function.
    """
    from sklearn.utils import resample

    # Separate majority and minority classes
    df_majority = df[df[target_column] == 0]  # Survival (majority)
    df_minority = df[df[target_column] == 1]  # Death (minority)

    # Undersample majority class to match minority class
    df_majority_undersampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    # Combine and shuffle
    df_balanced = pd.concat([df_majority_undersampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


def load_physionet_clean(split='test', n_samples=None):
    """
    Load clean PhysioNet dataset without any missingness.

    Args:
        split: 'train' or 'test' - splits data 80/20
        n_samples: Number of samples to load from the split

    Returns:
        X: Features as numpy array (n_samples, n_features)
        y: Labels as numpy array (n_samples,)
    """
    # Load the pre-balanced CSV file
    csv_file = PROJECT_ROOT / 'experiments' / 'tabular' / 'balanced_physionet_dataset_0-30.csv'

    if not csv_file.exists():
        raise FileNotFoundError(f"PhysioNet data file not found at {csv_file}")

    df_balanced = pd.read_csv(csv_file)

    # Split train/test (80/20)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['In-hospital_death'])

    df_split = train_df if split == 'train' else test_df

    # Sample if needed
    if n_samples is not None and n_samples < len(df_split):
        df_split = df_split.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Split into features and labels
    X = df_split.drop('In-hospital_death', axis=1).values
    y = df_split['In-hospital_death'].values

    return X, y


def load_physionet_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load PhysioNet with probabilistic feature missingness (binomial).

    Args:
        split: 'train' or 'test'
        p_ablate: Probability of ablating each feature (0.0 to 1.0)
        n_samples: Number of samples to load

    Returns:
        X: Features with probabilistic missingness (n_samples, n_features)
        y: Labels as numpy array (n_samples,)
    """
    X_clean, y = load_physionet_clean(split, n_samples)

    # Apply probabilistic missingness
    X_ablated = X_clean.copy()
    n_samples_actual, n_features = X_ablated.shape

    for i in range(n_samples_actual):
        # Each feature has p_ablate probability of being masked (binomial)
        mask = np.random.random(n_features) < p_ablate
        X_ablated[i, mask] = 0

    return X_ablated, y


def load_physionet_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load PhysioNet with exact fraction of features ablated.

    Args:
        split: 'train' or 'test'
        fraction_ablate: Exact fraction of features to ablate (0.0 to 1.0)
        n_samples: Number of samples to load

    Returns:
        X: Features with exact missingness (n_samples, n_features)
        y: Labels as numpy array (n_samples,)
    """
    X_clean, y = load_physionet_clean(split, n_samples)

    # Apply exact missingness
    X_ablated = X_clean.copy()
    n_samples_actual, n_features = X_ablated.shape
    n_to_ablate = int(n_features * fraction_ablate)

    for i in range(n_samples_actual):
        if n_to_ablate > 0:
            # Randomly select exact number of features to mask
            ablate_indices = np.random.choice(n_features, n_to_ablate, replace=False)
            X_ablated[i, ablate_indices] = 0

    return X_ablated, y


def load_physionet_clean_and_ablated(split='test', p_ablate=0.5, n_samples=None, seed=None, shuffle=False):
    """
    Load clean PhysioNet data and ablated versions with probabilistic feature masking.

    Ensures clean and ablated data are perfectly aligned (same samples).

    Args:
        split: 'train' or 'test'
        p_ablate: Probability of masking each feature (default 0.5)
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the data before returning (default False)

    Returns:
        X_clean: Clean features (n_samples, n_features)
        X_ablated: Ablated features (n_samples, n_features)
        y: Labels as numpy array (n_samples,)
    """
    # Load clean data
    X_clean, y = load_physionet_clean(split, n_samples)

    # Shuffle if requested
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X_clean))
        X_clean = X_clean[perm]
        y = y[perm]

    # Create ablated version with probabilistic feature masking
    X_ablated = X_clean.copy()
    n_samples_actual, n_features = X_ablated.shape

    if seed is not None:
        np.random.seed(seed)

    for i in range(n_samples_actual):
        # Probabilistic masking - each feature independently masked with prob p_ablate
        mask = np.random.random(n_features) < p_ablate
        X_ablated[i, mask] = 0

    return X_clean, X_ablated, y


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _dataset_to_tensors(dataset):
    """
    Convert a PyTorch dataset to tensors.
    """
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_images = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Loading data"):
        all_images.append(images)
        all_labels.append(labels)

    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return images, labels





def _mask_random_words_prob(text, mask_prob=0.15, replacement_token='UNKWORDZ', seed=None):
    """Replace random words/tokens with a replacement token."""
    if seed is not None:
        random.seed(seed)
    tokens = text.split()
    modified_tokens = [replacement_token if random.random() < mask_prob else token for token in tokens]
    return ' '.join(modified_tokens)


def _mask_random_words_exact(text, mask_prob=0.15, replacement_token='UNKWORDZ', seed=None):
    """Replace exactly mask_prob fraction of words/tokens with a replacement token."""
    if seed is not None:
        random.seed(seed)
    tokens = text.split()

    if len(tokens) == 0:
        return text

    # Calculate number of tokens to replace, ensuring it's within valid bounds
    num_to_replace = max(0, min(len(tokens), int(len(tokens) * mask_prob)))

    if num_to_replace == 0:
        return text

    replace_indices = random.sample(range(len(tokens)), num_to_replace)
    modified_tokens = [replacement_token if i in replace_indices else token for i, token in enumerate(tokens)]

    return ' '.join(modified_tokens)


def tokenize_and_mask_medical_qa(example, dataset_type, tokenizer=None, return_prompt=False):
    """
    Tokenizes the input text and creates a labels column for completion-only loss.
    This version constructs the final token sequence manually to ensure
    that the prompt/completion split is perfectly accurate.
    """

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # 1. Format the prompt and completion strings
    if dataset_type == "medqa":
        question = example['question']
        answer_key = example['answer_idx']
        options_dict = {opt['key']: opt['value'] for opt in example['options']}
        
        options_text = ""
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in options_dict:
                options_text += f"{letter}. {options_dict[letter]}\n"
    
    elif dataset_type == "medmcqa":
        question = example['question']
        correct_option_index = int(example['cop'])
        answer_key = ['A', 'B', 'C', 'D'][correct_option_index]
        options_text = (
            f"A. {example['opa']}\n"
            f"B. {example['opb']}\n"
            f"C. {example['opc']}\n"
            f"D. {example['opd']}\n"
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Note the space at the end of the prompt is important for some tokenizers
    prompt = f"Question: {question}\n\nOptions:\n{options_text}\nAnswer: "
    completion = f"{answer_key}"
    
    # 2. Tokenize parts separately
    # add_special_tokens=False is crucial to prevent tokenizer from adding BOS/EOS tokens in the middle
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    completion_tokens = tokenizer(completion, add_special_tokens=False)

    # 3. Manually construct the full token sequences
    input_ids = prompt_tokens['input_ids'] + completion_tokens['input_ids']
    attention_mask = prompt_tokens['attention_mask'] + completion_tokens['attention_mask']
    
    # 4. Manually construct the labels array
    # We mask the prompt tokens with -100 and use the completion tokens as labels
    labels = ([-100] * len(prompt_tokens['input_ids'])) + completion_tokens['input_ids']
    
    # The SFTTrainer expects these specific column names
    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if return_prompt:
        output['prompt'] = prompt
        output['completion'] = completion

    return output


def tokenize_and_mask_medqa(example, tokenizer=None):
    return tokenize_and_mask_medical_qa(example, "medqa", tokenizer)


def tokenize_and_mask_medmcqa(example, tokenizer=None):
    return tokenize_and_mask_medical_qa(example, "medmcqa", tokenizer)