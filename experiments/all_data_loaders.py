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
import pandas as pd
from PIL import Image
import datasets as huggingface_datasets
from transformers import AutoTokenizer


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
        # Use deterministic subset - take first n_samples
        indices = list(range(min(n_samples, len(dataset))))
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