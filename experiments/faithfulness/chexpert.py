#!/usr/bin/env python3
"""
Run Faithfulness, Deletion, and Insertion Experiments for MCal on CheXpert Dataset

CheXpert - Chest X-ray Classification (Binary: Cardiomegaly)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Add parent directory to path

# Import MCal components
from experiments.all_data_loaders import load_chexpert_clean
from mcal.calibrators.mcal_ce import SimpleMCalCE
from mcal.paths import MODEL_ROOT
from experiments.explanations import ImageKernelSHAP

# Import faithfulness metrics
from experiments.faithfulness_metrics import (
    ImageFaithfulnessPearson, ImageDeletionMetric, ImageInsertionMetric
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Configuration
config = {
    'n_train': 300,  # For MCal training
    'n_test': 50,    # Test samples for faithfulness evaluation
    'ablation_rate': 0.5,
    'patch_size': 16,
    'mcal_steps': 2000,
    'shap_samples': 1000,
    'n_deletion_steps': 10,
    'n_insertion_steps': 10,
    'batch_size': 32,
    'n_classes': 2  # Binary: Cardiomegaly (0 or 1)
}

print("\n" + "="*70)
print("FAITHFULNESS EXPERIMENTS FOR MCAL - CHEXPERT DATASET")
print("="*70)
print("\nConfiguration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# ============================================================================
# Load CheXpert Model and Data
# ============================================================================
print("\n" + "="*70)
print("1. LOADING CHEXPERT MODEL AND DATA")
print("="*70)

model_path = MODEL_ROOT / 'vit_timm_vanilla_chexpert_ps64_85tr76te.pth'

print(f"Loading model from: {model_path}")

# Create ViT model with 2 classes
base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=config['n_classes'])
state_dict = torch.load(model_path, map_location=device)
base_model.load_state_dict(state_dict)
base_model = base_model.to(device)
base_model.eval()

print("✓ Loaded CheXpert model (2 classes: Cardiomegaly binary)")

# Load datasets
print("Loading CheXpert datasets...")
train_images, train_labels = load_chexpert_clean('train')
test_images, test_labels = load_chexpert_clean('test')

print(f"✓ Loaded datasets:")
print(f"  Train: {train_images.shape[0]} images")
print(f"  Test: {test_images.shape[0]} images")
print(f"  Classes: {config['n_classes']}")
print(f"  Test class distribution: {np.bincount(test_labels.numpy())}")

# Subsample for experiments
train_images = train_images[:config['n_train']]
train_labels = train_labels[:config['n_train']]
test_images = test_images[:config['n_test']]
test_labels = test_labels[:config['n_test']]

print(f"\nAfter subsampling:")
print(f"  Train: {train_images.shape[0]} images")
print(f"  Test: {test_images.shape[0]} images")
print(f"  Test class distribution: {np.bincount(test_labels.numpy())}")

# Move to device
train_images = train_images.to(device)
train_labels = train_labels.to(device)
test_images = test_images.to(device)
test_labels = test_labels.to(device)

# ============================================================================
# Evaluate Base Model
# ============================================================================
print("\n" + "="*70)
print("2. EVALUATING BASE MODEL")
print("="*70)

with torch.no_grad():
    train_preds = []
    for i in range(0, len(train_images), config['batch_size']):
        batch = train_images[i:i+config['batch_size']]
        preds = base_model(batch)
        train_preds.append(preds)
    train_preds = torch.cat(train_preds, dim=0)

    test_preds = []
    for i in range(0, len(test_images), config['batch_size']):
        batch = test_images[i:i+config['batch_size']]
        preds = base_model(batch)
        test_preds.append(preds)
    test_preds = torch.cat(test_preds, dim=0)

train_acc = (train_preds.argmax(dim=1) == train_labels).float().mean()
test_acc = (test_preds.argmax(dim=1) == test_labels).float().mean()

print(f"✓ Base model accuracy:")
print(f"  Train: {train_acc:.3f}")
print(f"  Test: {test_acc:.3f}")

# ============================================================================
# Create Ablated Training Data and Train MCal
# ============================================================================
print("\n" + "="*70)
print("3. TRAINING MCAL CALIBRATOR")
print("="*70)

# Create ablated training images (patch ablation)
print("Creating ablated training data...")
ablated_train_images = train_images.clone()

patch_size = config['patch_size']
C, H, W = train_images.shape[1:]
n_patches_h, n_patches_w = H // patch_size, W // patch_size

for i in range(len(ablated_train_images)):
    # Random patch mask
    patch_mask = torch.rand(n_patches_h, n_patches_w, device=device) < config['ablation_rate']
    # Upsample to image size
    patch_mask = torch.nn.functional.interpolate(
        patch_mask.float().unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='nearest'
    ).squeeze(0).squeeze(0).bool()
    # Apply mask (set to 0)
    ablated_train_images[i, :, patch_mask] = 0.0

print(f"✓ Ablated {config['ablation_rate']*100:.0f}% of patches per image")

# Get predictions on ablated data
print("Computing predictions on ablated training data...")
with torch.no_grad():
    ablated_preds = []
    for i in range(0, len(ablated_train_images), config['batch_size']):
        batch = ablated_train_images[i:i+config['batch_size']]
        preds = base_model(batch)
        ablated_preds.append(preds)
    ablated_preds = torch.cat(ablated_preds, dim=0)

# Train MCal
print("\nTraining MCal calibrator...")
calibrator = SimpleMCalCE(num_classes=config['n_classes']).to(device)
stats = calibrator.fit(
    ablated_logits=ablated_preds,
    target_labels=train_labels,
    max_steps=config['mcal_steps'],
    verbose=True
)

print(f"\n✓ MCal training complete!")
print(f"  Final Loss: {stats['loss'][-1]:.4f}")
print(f"  Final Accuracy: {stats['acc'][-1]:.3f}")

# Create calibrated model wrapper
class CalibratedVisionModel(nn.Module):
    def __init__(self, base_model, calibrator):
        super().__init__()
        self.base_model = base_model
        self.calibrator = calibrator

    def forward(self, x):
        logits = self.base_model(x)
        calibrated_logits = self.calibrator(logits)
        return calibrated_logits

calib_model = CalibratedVisionModel(base_model, calibrator).to(device)
calib_model.eval()

print("✓ Created calibrated model")

# ============================================================================
# Generate SHAP Explanations
# ============================================================================
print("\n" + "="*70)
print("4. GENERATING SHAP EXPLANATIONS")
print("="*70)

print(f"Creating SHAP explainers (samples: {config['shap_samples']})...")

# Create explainers
explainer_uncal = ImageKernelSHAP(
    model=base_model,
    num_samples=config['shap_samples'],
    patch_size=config['patch_size']
)
explainer_calib = ImageKernelSHAP(
    model=calib_model,
    num_samples=config['shap_samples'],
    patch_size=config['patch_size']
)

# Generate SHAP values for test images
print(f"Computing SHAP values for {len(test_images)} test samples...")
uncal_shap_values = []
calib_shap_values = []

for i in tqdm(range(len(test_images)), desc="SHAP explanations"):
    img = test_images[i]  # Shape: (C, H, W)
    label = test_labels[i].item()

    shap_uncal = explainer_uncal.explain_instance(img, label=label)
    shap_calib = explainer_calib.explain_instance(img, label=label)

    uncal_shap_values.append(shap_uncal.cpu())
    calib_shap_values.append(shap_calib.cpu())

print("✓ SHAP explanations generated")

# ============================================================================
# Compute Faithfulness Metrics
# ============================================================================
print("\n" + "="*70)
print("5. COMPUTING FAITHFULNESS METRICS")
print("="*70)

# Initialize metrics
faithfulness_metric = ImageFaithfulnessPearson(baseline_value=0.0)
deletion_metric = ImageDeletionMetric(baseline_value=0.0, n_steps=config['n_deletion_steps'])
insertion_metric = ImageInsertionMetric(baseline_value=0.0, n_steps=config['n_insertion_steps'])

# Storage for results
results = {
    'uncalibrated': {'faithfulness': [], 'deletion_auc': [], 'insertion_auc': []},
    'calibrated': {'faithfulness': [], 'deletion_auc': [], 'insertion_auc': []},
    'deletion_curves_uncal': [],
    'deletion_curves_cal': [],
    'insertion_curves_uncal': [],
    'insertion_curves_cal': []
}

print(f"Computing metrics for {len(test_images)} test samples...")
for i in tqdm(range(len(test_images)), desc="Computing metrics"):
    img = test_images[i]
    label = test_labels[i].item()

    uncal_attrs = uncal_shap_values[i]
    calib_attrs = calib_shap_values[i]

    # Uncalibrated metrics
    faith_uncal = faithfulness_metric.compute(base_model, img, uncal_attrs, label)
    del_uncal = deletion_metric.compute(base_model, img, uncal_attrs, label)
    ins_uncal = insertion_metric.compute(base_model, img, uncal_attrs, label)

    results['uncalibrated']['faithfulness'].append(faith_uncal)
    results['uncalibrated']['deletion_auc'].append(del_uncal['auc'])
    results['uncalibrated']['insertion_auc'].append(ins_uncal['auc'])
    results['deletion_curves_uncal'].append((del_uncal['fractions'], del_uncal['scores']))
    results['insertion_curves_uncal'].append((ins_uncal['fractions'], ins_uncal['scores']))

    # Calibrated metrics
    faith_cal = faithfulness_metric.compute(calib_model, img, calib_attrs, label)
    del_cal = deletion_metric.compute(calib_model, img, calib_attrs, label)
    ins_cal = insertion_metric.compute(calib_model, img, calib_attrs, label)

    results['calibrated']['faithfulness'].append(faith_cal)
    results['calibrated']['deletion_auc'].append(del_cal['auc'])
    results['calibrated']['insertion_auc'].append(ins_cal['auc'])
    results['deletion_curves_cal'].append((del_cal['fractions'], del_cal['scores']))
    results['insertion_curves_cal'].append((ins_cal['fractions'], ins_cal['scores']))

print("\n✓ Metrics computed!")

# ============================================================================
# Results Summary
# ============================================================================
print("\n" + "="*70)
print("6. RESULTS SUMMARY")
print("="*70)

summary = pd.DataFrame({
    'Metric': ['Faithfulness (Pearson ρ) ↑', 'Deletion AUC ↓', 'Insertion AUC ↑'],
    'Uncalibrated': [
        np.mean(results['uncalibrated']['faithfulness']),
        np.mean(results['uncalibrated']['deletion_auc']),
        np.mean(results['uncalibrated']['insertion_auc'])
    ],
    'MCal Calibrated': [
        np.mean(results['calibrated']['faithfulness']),
        np.mean(results['calibrated']['deletion_auc']),
        np.mean(results['calibrated']['insertion_auc'])
    ]
})

improvements = []
for i in range(len(summary)):
    uncal = summary.iloc[i]['Uncalibrated']
    cal = summary.iloc[i]['MCal Calibrated']
    metric_name = summary.iloc[i]['Metric']

    if 'Deletion' in metric_name:
        imp = ((uncal - cal) / abs(uncal)) * 100 if uncal != 0 else 0
    else:
        imp = ((cal - uncal) / abs(uncal)) * 100 if uncal != 0 else 0
    improvements.append(f"{imp:+.1f}%")

summary['Improvement'] = improvements

print("\n" + "="*70)
print(f"CHEXPERT RESULTS (averaged over {len(test_images)} test samples)")
print("="*70)
print(summary.to_string(index=False))
print("="*70)

# ============================================================================
# Create Visualizations
# ============================================================================
print("\n" + "="*70)
print("7. CREATING VISUALIZATIONS")
print("="*70)

output_dir = Path(__file__).resolve().parents[1] / 'results'
output_dir.mkdir(exist_ok=True)

# Deletion Curves
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for fracs, scores in results['deletion_curves_uncal']:
    ax.plot(fracs, scores.cpu().numpy() if torch.is_tensor(scores) else scores,
            color='steelblue', alpha=0.3, linewidth=1)
for fracs, scores in results['deletion_curves_cal']:
    ax.plot(fracs, scores.cpu().numpy() if torch.is_tensor(scores) else scores,
            color='darkorange', alpha=0.3, linewidth=1)

avg_fracs = results['deletion_curves_uncal'][0][0]
avg_uncal = np.mean([scores.cpu().numpy() if torch.is_tensor(scores) else scores
                      for _, scores in results['deletion_curves_uncal']], axis=0)
avg_cal = np.mean([scores.cpu().numpy() if torch.is_tensor(scores) else scores
                    for _, scores in results['deletion_curves_cal']], axis=0)
ax.plot(avg_fracs, avg_uncal, color='steelblue', linewidth=3, label='Uncalibrated', marker='o')
ax.plot(avg_fracs, avg_cal, color='darkorange', linewidth=3, label='MCal Calibrated', marker='s')

ax.set_xlabel('Fraction of Patches Deleted', fontsize=12)
ax.set_ylabel('Prediction Score (True Class)', fontsize=12)
ax.set_title('CheXpert: Deletion Curves (MCal vs Uncalibrated)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'chexpert_deletion_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'chexpert_deletion_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved deletion curves")

# Insertion Curves
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for fracs, scores in results['insertion_curves_uncal']:
    ax.plot(fracs, scores.cpu().numpy() if torch.is_tensor(scores) else scores,
            color='steelblue', alpha=0.3, linewidth=1)
for fracs, scores in results['insertion_curves_cal']:
    ax.plot(fracs, scores.cpu().numpy() if torch.is_tensor(scores) else scores,
            color='darkorange', alpha=0.3, linewidth=1)

avg_fracs = results['insertion_curves_uncal'][0][0]
avg_uncal = np.mean([scores.cpu().numpy() if torch.is_tensor(scores) else scores
                      for _, scores in results['insertion_curves_uncal']], axis=0)
avg_cal = np.mean([scores.cpu().numpy() if torch.is_tensor(scores) else scores
                    for _, scores in results['insertion_curves_cal']], axis=0)
ax.plot(avg_fracs, avg_uncal, color='steelblue', linewidth=3, label='Uncalibrated', marker='o')
ax.plot(avg_fracs, avg_cal, color='darkorange', linewidth=3, label='MCal Calibrated', marker='s')

ax.set_xlabel('Fraction of Patches Inserted', fontsize=12)
ax.set_ylabel('Prediction Score (True Class)', fontsize=12)
ax.set_title('CheXpert: Insertion Curves (MCal vs Uncalibrated)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'chexpert_insertion_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'chexpert_insertion_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved insertion curves")

# Faithfulness Bar Chart
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics = ['Faithfulness\n(Pearson ρ)', 'Deletion\nAUC', 'Insertion\nAUC']
uncal_vals = [
    np.mean(results['uncalibrated']['faithfulness']),
    np.mean(results['uncalibrated']['deletion_auc']),
    np.mean(results['uncalibrated']['insertion_auc'])
]
cal_vals = [
    np.mean(results['calibrated']['faithfulness']),
    np.mean(results['calibrated']['deletion_auc']),
    np.mean(results['calibrated']['insertion_auc'])
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, uncal_vals, width, label='Uncalibrated', color='steelblue')
bars2 = ax.bar(x + width/2, cal_vals, width, label='MCal Calibrated', color='darkorange')

ax.set_ylabel('Score', fontsize=13)
ax.set_title('CheXpert: Faithfulness Metrics Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'chexpert_faithfulness_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'chexpert_faithfulness_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved faithfulness comparison")

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*70)
print("8. SAVING RESULTS")
print("="*70)

summary.to_csv(output_dir / 'chexpert_faithfulness_results_shap.csv', index=False)
print(f"✓ Saved results to {output_dir / 'chexpert_faithfulness_results_shap.csv'}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT COMPLETE!")
print("="*70)
print(f"\n✅ KEY FINDINGS (CheXpert - Cardiomegaly Binary Classification):")
print(f"  Faithfulness (Pearson ρ): {np.mean(results['uncalibrated']['faithfulness']):.3f} → {np.mean(results['calibrated']['faithfulness']):.3f}")
print(f"  Deletion AUC:             {np.mean(results['uncalibrated']['deletion_auc']):.3f} → {np.mean(results['calibrated']['deletion_auc']):.3f}")
print(f"  Insertion AUC:            {np.mean(results['uncalibrated']['insertion_auc']):.3f} → {np.mean(results['calibrated']['insertion_auc']):.3f}")
print(f"\n📁 Output files saved to results/")
print("\n" + "="*70)
