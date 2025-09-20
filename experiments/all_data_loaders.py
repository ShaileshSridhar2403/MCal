#!/usr/bin/env python3
"""
Unified Data Loader for MCal Experiments

A single file containing all data loading functions with a consistent API.
Just loads raw data - no models, no predictions, just data.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets
import random
from tqdm import tqdm
import json
import pandas as pd
from PIL import Image
import datasets as huggingface_datasets

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.augmentation.patch_cutout import PatchCutout

# Default data directory
DATA_ROOT = PROJECT_ROOT / "data"


# =============================================================================
# STANDALONE UTILITY FUNCTIONS (previously imported from utils)
# =============================================================================


def mask_random_words(text, removal_fraction=0.15, replacement_token='[MASK]'):
    """Replace random words/tokens with a replacement token."""
    tokens = text.split()
    num_replace = int(len(tokens) * removal_fraction)

    if num_replace > 0 and len(tokens) > 0:
        indices_to_replace = random.sample(range(len(tokens)), min(num_replace, len(tokens)))
    else:
        indices_to_replace = []

    modified_tokens = [replacement_token if i in indices_to_replace else token for i, token in enumerate(tokens)]
    modified_text = ' '.join(modified_tokens)

    return modified_text



# =============================================================================
# MRI DATASET
# =============================================================================

def load_mri_clean(split='test', n_samples=None):
    """
    Load clean MRI dataset without any augmentation.

    Args:
        split: 'train' or 'test'
        n_samples: Number of samples to load (None = all)

    Returns:
        images: (n_samples, 3, 224, 224) tensor
        labels: (n_samples,) tensor
    """
    # MRI data is in vision/data/Training and vision/data/Testing
    if split == 'train':
        data_dir = Path(__file__).parent / "vision" / "data" / "Training"
    else:
        data_dir = Path(__file__).parent / "vision" / "data" / "Testing"

    # Create transforms - just resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)

    if n_samples:
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        dataset = Subset(dataset, indices)

    return _dataset_to_tensors(dataset)


def load_mri_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load MRI with probabilistic patch ablation.

    Args:
        split: 'train' or 'test'
        p_ablate: Probability of ablating each patch (0.0 to 1.0)
        n_samples: Number of samples to load

    Returns:
        images: (n_samples, 3, 224, 224) tensor with ablation applied
        labels: (n_samples,) tensor
    """
    images, labels = load_mri_clean(split, n_samples)

    # Apply PatchCutout with probability p
    augmenter = PatchCutout(
        patch_height=56,
        patch_width=56,
        removal_fraction=p_ablate,
        random_removal_fraction=False,
        fill_val=0
    )

    ablated_images = torch.stack([augmenter(img) for img in images])
    return ablated_images, labels


def load_mri_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load MRI with exact fraction of patches ablated.

    Args:
        split: 'train' or 'test'
        fraction_ablate: Exact fraction of patches to ablate (0.0 to 1.0)
        n_samples: Number of samples to load

    Returns:
        images: (n_samples, 3, 224, 224) tensor with ablation applied
        labels: (n_samples,) tensor
    """
    # Same as probabilistic since PatchCutout with random_removal_fraction=False gives exact fraction
    return load_mri_ablated_prob(split, fraction_ablate, n_samples)


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

    # Sample if needed
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

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
    # BreakHis data is in vision/data/BreakHis/
    if split == 'train':
        data_dir = Path(__file__).parent / "vision" / "data" / "BreakHis" / "BreakHisTraining"
    else:
        data_dir = Path(__file__).parent / "vision" / "data" / "BreakHis" / "BreakHisTesting"

    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)

    if n_samples:
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
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

    Returns:
        texts: List of (question, options) tuples
        labels: numpy array of correct answer indices
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

    # Convert to list for sampling
    dataset_list = list(dataset)

    # Apply sampling if needed
    if n_samples and len(dataset_list) > n_samples:
        random.shuffle(dataset_list)
        dataset_list = dataset_list[:n_samples]
    elif n_samples is None:
        # Default to 1000 samples if not specified
        if len(dataset_list) > 1000:
            random.shuffle(dataset_list)
            dataset_list = dataset_list[:1000]

    texts = []
    labels = []

    for item in dataset_list:
        # Convert options list to dict format
        options_dict = {}
        for opt in item['options']:
            options_dict[opt['key']] = opt['value']

        texts.append((item['question'], options_dict))
        # Convert answer letter to index (A=0, B=1, C=2, D=3, E=4)
        labels.append(ord(item['answer_idx']) - ord('A'))

    return texts, torch.tensor(labels, dtype=torch.long)


def load_medqa_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load MedQA with probabilistic token removal.

    Returns:
        texts: List of (modified_question, options) tuples
        labels: numpy array of correct answer indices
    """
    texts, labels = load_medqa_clean(split, n_samples)

    modified_texts = []
    for question, options in texts:
        modified_question = mask_random_words(
            question,
            removal_fraction=p_ablate,
            replacement_token='[MASK]'
        )
        modified_texts.append((modified_question, options))

    return modified_texts, labels


def load_medqa_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load MedQA with exact fraction of tokens removed.
    """
    return load_medqa_ablated_prob(split, fraction_ablate, n_samples)


def load_medqa_fractionwise(split='test', n_fractions=16, n_samples=None):
    """
    Load MedQA with multiple ablation fractions.

    Returns:
        texts: List of n_fractions lists, each containing (question, options) tuples
        labels: numpy array of correct answer indices
    """
    clean_texts, labels = load_medqa_clean(split, n_samples)

    all_ablated_texts = []

    for i in range(n_fractions):
        fraction = i / n_fractions
        if fraction == 0:
            all_ablated_texts.append(clean_texts)
        else:
            modified_texts = []
            for question, options in clean_texts:
                modified_question = mask_random_words(
                    question,
                    removal_fraction=fraction,
                    replacement_token='[MASK]'
                )
                modified_texts.append((modified_question, options))
            all_ablated_texts.append(modified_texts)

    return all_ablated_texts, labels


# =============================================================================
# MEDMCQA DATASET
# =============================================================================

def load_medmcqa_clean(split='test', n_samples=None):
    """
    Load clean MedMCQA dataset without any text manipulation.

    Returns:
        texts: List of (question, options_dict) tuples
        labels: numpy array of correct answer indices (0-indexed)
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

    # Convert to list for processing
    dataset_list = list(dataset)

    # Apply sampling
    if n_samples and len(dataset_list) > n_samples:
        random.shuffle(dataset_list)
        dataset_list = dataset_list[:n_samples]

    texts = []
    labels = []

    for item in dataset_list:
        # HuggingFace format has: 'question', 'opa', 'opb', 'opc', 'opd', 'cop'
        # Create options dict for consistency with MedQA format
        options = {
            'A': item['opa'],
            'B': item['opb'],
            'C': item['opc'],
            'D': item['opd']
        }
        texts.append((item['question'], options))
        labels.append(item['cop'])  # 0-indexed: 0=A, 1=B, 2=C, 3=D

    return texts, torch.tensor(labels, dtype=torch.long)


def load_medmcqa_ablated_prob(split='test', p_ablate=0.5, n_samples=None):
    """
    Load MedMCQA with probabilistic token removal.
    """
    texts, labels = load_medmcqa_clean(split, n_samples)

    modified_texts = []
    for question, options in texts:
        modified_question = mask_random_words(
            question,
            removal_fraction=p_ablate,
            replacement_token='[MASK]'
        )
        modified_texts.append((modified_question, options))

    return modified_texts, labels


def load_medmcqa_ablated_exact(split='test', fraction_ablate=0.5, n_samples=None):
    """
    Load MedMCQA with exact fraction of tokens removed.
    """
    return load_medmcqa_ablated_prob(split, fraction_ablate, n_samples)


def load_medmcqa_fractionwise(split='test', n_fractions=16, n_samples=None):
    """
    Load MedMCQA with multiple ablation fractions.
    """
    clean_texts, labels = load_medmcqa_clean(split, n_samples)

    all_ablated_texts = []

    for i in range(n_fractions):
        fraction = i / n_fractions
        if fraction == 0:
            all_ablated_texts.append(clean_texts)
        else:
            modified_texts = []
            for question, options in clean_texts:
                modified_question = mask_random_words(
                    question,
                    removal_fraction=fraction,
                    replacement_token='[MASK]'
                )
                modified_texts.append((modified_question, options))
            all_ablated_texts.append(modified_texts)

    return all_ablated_texts, labels


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

