#!/usr/bin/env python3
"""
Download and prepare MedMCQA dataset for MCal experiments.
"""

import json
import random
from datasets import load_dataset
from pathlib import Path

def download_and_prepare_medmcqa():
    """Download MedMCQA and create balanced dev set."""
    print("Loading MedMCQA dataset from Hugging Face...")

    # Load the dataset
    dataset = load_dataset("medmcqa")
    dev_data = dataset['validation']

    print(f"Total validation samples: {len(dev_data)}")

    # Group by answer choice (cop field: 1=A, 2=B, 3=C, 4=D)
    answer_groups = {1: [], 2: [], 3: [], 4: []}

    for item in dev_data:
        cop = item['cop']
        if cop in answer_groups:
            answer_groups[cop].append(item)

    print("Distribution by answer choice:")
    for cop, items in answer_groups.items():
        print(f"  {cop} ({chr(64+cop)}): {len(items)} samples")

    # Create balanced dataset - take equal samples from each group
    min_count = min(len(items) for items in answer_groups.values())
    balanced_samples = []

    for cop, items in answer_groups.items():
        # Randomly sample min_count items
        random.seed(42)  # For reproducibility
        sampled = random.sample(items, min_count)
        balanced_samples.extend(sampled)

    # Shuffle the final balanced dataset
    random.seed(42)
    random.shuffle(balanced_samples)

    print(f"Created balanced dataset with {len(balanced_samples)} samples")
    print(f"({min_count} samples per answer choice)")

    # Save to JSON
    output_dir = Path("dataset_store/language/medmcqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dev_balanced.json"
    with open(output_file, 'w') as f:
        json.dump(balanced_samples, f, indent=2)

    print(f"Saved balanced dataset to: {output_file}")

    # Also save a sample for quick testing
    sample_file = output_dir / "dev_sample.json"
    sample_data = balanced_samples[:100]  # First 100 samples
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"Saved sample dataset (100 items) to: {sample_file}")

if __name__ == "__main__":
    download_and_prepare_medmcqa()