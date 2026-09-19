#!/usr/bin/env python3
"""
Analyze MRI Confidence Scores: Calibrated vs Uncalibrated
Shows pairwise confidence comparison ranked by uncalibrated confidence
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path

from experiments.all_data_loaders import load_mri_clean
from mcal.configs.model_dict import MODEL_DICT
import timm

# Add src to path for MCal imports
from mcal.calibrators.mcal_ce import SimpleMCalCE

# Configuration
config = {
    'n_train': 300,
    'n_test': 50,
    'mcal_steps': 2000,
    'batch_size': 32,
    'n_classes': 4
}

print("="*70)
print("MRI PAIRWISE CONFIDENCE ANALYSIS")
print("="*70)

# Load data
print("\n1. Loading MRI data...")
train_dataset = load_mri_clean(split='train')
test_dataset = load_mri_clean(split='test')

# Convert to tensors and limit samples
print("Converting datasets to tensors...")
train_images = []
train_labels = []
for i, (img, label) in enumerate(tqdm(train_dataset, desc="Loading train")):
    if i >= config['n_train']:
        break
    train_images.append(img)
    train_labels.append(label)
train_images = torch.stack(train_images)
train_labels = torch.tensor(train_labels)

test_images = []
test_labels = []
for i, (img, label) in enumerate(tqdm(test_dataset, desc="Loading test")):
    if i >= config['n_test']:
        break
    test_images.append(img)
    test_labels.append(label)
test_images = torch.stack(test_images)
test_labels = torch.tensor(test_labels)

print(f"Train: {len(train_images)} images")
print(f"Test: {len(test_images)} images")
print(f"Classes: {config['n_classes']}")

# Load model
print("\n2. Loading model...")
device = torch.device('cpu')
base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=config['n_classes'])

model_path = Path(__file__).parent.parent / 'saved_models' / MODEL_DICT['mri']['vanilla']
checkpoint = torch.load(model_path, map_location=device)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['state_dict'])
    else:
        base_model.load_state_dict(checkpoint)
else:
    base_model.load_state_dict(checkpoint)

base_model.eval()
print(f"✓ Loaded model from {model_path}")

# Train MCal calibrator
print(f"\n3. Training MCal calibrator...")
print(f"MCal steps: {config['mcal_steps']}")

# Get train logits
train_logits = []
with torch.no_grad():
    for i in tqdm(range(0, len(train_images), config['batch_size']), desc="Train logits"):
        batch = train_images[i:i+config['batch_size']]
        outputs = base_model(batch)
        train_logits.append(outputs)
train_logits = torch.cat(train_logits, dim=0)

print(f"Train logits shape: {train_logits.shape}")

# Train calibrator with correct API
calibrator = SimpleMCalCE(num_classes=config['n_classes'])

calibrator.fit(
    ablated_logits=train_logits,
    target_labels=train_labels,
    max_steps=config['mcal_steps'],
    lr=1e-3,
    verbose=True
)

print("✓ MCal training complete!")

# Create calibrated model wrapper
class CalibratedVisionModel(nn.Module):
    def __init__(self, base_model, calibrator):
        super().__init__()
        self.base_model = base_model
        self.calibrator = calibrator

    def forward(self, x):
        with torch.no_grad():
            logits = self.base_model(x)
            calibrated_logits = self.calibrator(logits)
        return calibrated_logits

calib_model = CalibratedVisionModel(base_model, calibrator)
calib_model.eval()

print("✓ Created calibrated model")

# Get predictions and confidence scores
print("\n4. Computing confidence scores for test samples...")

uncal_confidences = []
calib_confidences = []
predictions_uncal = []
predictions_calib = []
true_labels = []

with torch.no_grad():
    for i in tqdm(range(len(test_images)), desc="Computing scores"):
        img = test_images[i:i+1]
        label = test_labels[i].item()

        # Uncalibrated
        logits_uncal = base_model(img)
        probs_uncal = torch.softmax(logits_uncal, dim=1)[0]
        pred_uncal = probs_uncal.argmax().item()
        conf_uncal = probs_uncal.max().item()

        # Calibrated
        logits_calib = calib_model(img)
        probs_calib = torch.softmax(logits_calib, dim=1)[0]
        pred_calib = probs_calib.argmax().item()
        conf_calib = probs_calib.max().item()

        uncal_confidences.append(conf_uncal)
        calib_confidences.append(conf_calib)
        predictions_uncal.append(pred_uncal)
        predictions_calib.append(pred_calib)
        true_labels.append(label)

print("✓ Confidence scores computed!")

# Create DataFrame
df = pd.DataFrame({
    'sample_idx': range(len(test_images)),
    'true_label': true_labels,
    'pred_uncal': predictions_uncal,
    'pred_calib': predictions_calib,
    'conf_uncal': uncal_confidences,
    'conf_calib': calib_confidences,
    'conf_diff': np.array(calib_confidences) - np.array(uncal_confidences)
})

# Add correctness
df['correct_uncal'] = df['pred_uncal'] == df['true_label']
df['correct_calib'] = df['pred_calib'] == df['true_label']

# Rank by uncalibrated confidence
df['rank_uncal'] = df['conf_uncal'].rank(ascending=False, method='first').astype(int)
df['rank_calib'] = df['conf_calib'].rank(ascending=False, method='first').astype(int)

# Sort by uncalibrated rank
df_sorted = df.sort_values('rank_uncal').reset_index(drop=True)

print("\n" + "="*70)
print("5. PAIRWISE CONFIDENCE SCORES VS RANK")
print("="*70)

print("\nSorted by Uncalibrated Confidence (Highest to Lowest):")
print("="*70)

# Print header
print(f"{'Rank':<6} {'Sample':<8} {'Label':<7} {'Uncal Conf':<12} {'Calib Conf':<12} {'Diff':<10} {'Uncal ✓':<10} {'Calib ✓':<10}")
print("-"*80)

# Print each sample
for idx, row in df_sorted.iterrows():
    rank = row['rank_uncal']
    sample = row['sample_idx']
    label = row['true_label']
    conf_u = row['conf_uncal']
    conf_c = row['conf_calib']
    diff = row['conf_diff']
    correct_u = '✓' if row['correct_uncal'] else '✗'
    correct_c = '✓' if row['correct_calib'] else '✗'

    print(f"{rank:<6} {sample:<8} {label:<7} {conf_u:<12.4f} {conf_c:<12.4f} {diff:>+9.4f} {correct_u:<10} {correct_c:<10}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nMean Confidence:")
print(f"  Uncalibrated: {df['conf_uncal'].mean():.4f} ± {df['conf_uncal'].std():.4f}")
print(f"  Calibrated:   {df['conf_calib'].mean():.4f} ± {df['conf_calib'].std():.4f}")

print(f"\nMedian Confidence:")
print(f"  Uncalibrated: {df['conf_uncal'].median():.4f}")
print(f"  Calibrated:   {df['conf_calib'].median():.4f}")

print(f"\nAccuracy:")
print(f"  Uncalibrated: {df['correct_uncal'].sum()}/{len(df)} = {df['correct_uncal'].mean():.1%}")
print(f"  Calibrated:   {df['correct_calib'].sum()}/{len(df)} = {df['correct_calib'].mean():.1%}")

print(f"\nConfidence Change:")
print(f"  Mean Δ: {df['conf_diff'].mean():+.4f}")
print(f"  Median Δ: {df['conf_diff'].median():+.4f}")
print(f"  Std Δ: {df['conf_diff'].std():.4f}")
print(f"  Samples with increased confidence: {(df['conf_diff'] > 0).sum()}/{len(df)}")
print(f"  Samples with decreased confidence: {(df['conf_diff'] < 0).sum()}/{len(df)}")

# Rank correlation
from scipy.stats import spearmanr, pearsonr

spearman_corr, spearman_p = spearmanr(df['rank_uncal'], df['rank_calib'])
pearson_corr, pearson_p = pearsonr(df['conf_uncal'], df['conf_calib'])

print(f"\nRank Correlation:")
print(f"  Spearman ρ: {spearman_corr:.4f} (p={spearman_p:.4e})")
print(f"  Pearson r:  {pearson_corr:.4f} (p={pearson_p:.4e})")

# Save to CSV
output_file = Path('results/mri_confidence_pairwise.csv')
output_file.parent.mkdir(exist_ok=True)
df_sorted.to_csv(output_file, index=False)
print(f"\n✓ Saved detailed results to {output_file}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
