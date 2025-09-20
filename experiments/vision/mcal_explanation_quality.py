"""
MCal Explanation Quality Experiment for MRI Dataset

This experiment tests whether MCal calibration improves the quality of model explanations
by comparing sufficiency and comprehensiveness metrics on LIME explanations.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from typing import Dict
import argparse

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.all_data_loaders import load_mri_clean, load_mri_ablated_prob
from src.calibrators.mcal_ce import MCal_CE
from src.calibrators.mcal import MCal


def load_mri_model(model_path: str = None) -> nn.Module:
    """Load pre-trained MRI Vision Transformer model."""
    if model_path is None:
        # Look for model in project root
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'vit_timm_standard_mri_ps64_35e.pth'
        )

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Determine number of classes from saved weights
    num_classes = state_dict['head.weight'].shape[0]  # Should be 4

    # Create model with correct architecture
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


class CustomLIME:
    """Custom LIME implementation with square patches for image explanation."""

    def __init__(self, model: nn.Module, num_samples: int = 100):
        """
        Initialize Custom LIME explainer.

        Args:
            model: PyTorch model that outputs probabilities
            num_samples: Number of perturbed samples for LIME
        """
        self.model = model
        self.num_samples = num_samples

    def create_square_patches(self, image_size: int = 224, patch_size: int = 56):
        """
        Create 4x4 grid of square patches (16 total patches).

        Returns:
            masks: Binary masks for each patch (16, 224, 224)
        """
        n_patches_per_dim = image_size // patch_size
        n_patches = n_patches_per_dim * n_patches_per_dim

        masks = torch.zeros((n_patches, image_size, image_size))

        patch_idx = 0
        for i in range(n_patches_per_dim):
            for j in range(n_patches_per_dim):
                row_start = i * patch_size
                row_end = row_start + patch_size
                col_start = j * patch_size
                col_end = col_start + patch_size

                masks[patch_idx, row_start:row_end, col_start:col_end] = 1
                patch_idx += 1

        return masks

    def generate_perturbations(self, image: torch.Tensor, masks: torch.Tensor):
        """
        Generate perturbed samples by randomly masking patches.

        Args:
            image: Input image (3, 224, 224)
            masks: Binary masks for patches (16, 224, 224)

        Returns:
            perturbed_images: Perturbed samples (num_samples, 3, 224, 224)
            binary_samples: Binary matrix indicating which patches are kept (num_samples, 16)
        """
        n_patches = masks.shape[0]
        device = image.device

        # Generate random binary matrix for patch selection
        binary_samples = torch.randint(0, 2, size=(self.num_samples, n_patches))

        # Ensure at least one patch is kept in each sample
        for i in range(self.num_samples):
            if binary_samples[i].sum() == 0:
                binary_samples[i, torch.randint(0, n_patches, (1,)).item()] = 1

        # Generate perturbed images
        perturbed_images = []

        for i in range(self.num_samples):
            # Create combined mask for this sample
            combined_mask = torch.zeros((224, 224), device=device)
            for j in range(n_patches):
                if binary_samples[i, j] == 1:  # Keep this patch
                    combined_mask += masks[j].to(device)

            # Apply mask to all channels (mask out by setting to 0/black)
            masked_image = image.clone()
            for c in range(3):  # Apply to all RGB channels
                masked_image[c] = masked_image[c] * combined_mask

            perturbed_images.append(masked_image)

        perturbed_images = torch.stack(perturbed_images)

        return perturbed_images, binary_samples

    def explain_instance(self, image: torch.Tensor, label: int = None):
        """
        Generate LIME explanation for a single image.

        Args:
            image: Input image (3, 224, 224)
            label: Class to explain (if None, use predicted class)

        Returns:
            importance_scores: Importance score for each patch (16,)
        """
        # Move model to same device as image
        device = image.device

        # Create patches
        masks = self.create_square_patches()

        # Generate perturbations
        perturbed_images, binary_samples = self.generate_perturbations(image, masks)

        # Get predictions for perturbed samples
        with torch.no_grad():
            perturbed_images = perturbed_images.to(device)
            predictions = self.model(perturbed_images)

            # Get original prediction if label not specified
            if label is None:
                orig_pred = self.model(image.unsqueeze(0))
                label = orig_pred.argmax(dim=1).item()

            # Extract probabilities for target class
            probs = predictions[:, label].cpu()

        # Fit linear model using PyTorch
        # Weight samples by similarity (number of patches kept)
        weights = binary_samples.sum(dim=1).float() / binary_samples.shape[1]

        # Add bias term to binary samples
        X = torch.cat([binary_samples.float(), torch.ones(self.num_samples, 1)], dim=1)

        # Weighted least squares solution
        W = torch.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ probs

        # Add ridge regularization
        alpha = 1.0
        XtWX = XtWX + alpha * torch.eye(XtWX.shape[0])

        # Solve for coefficients
        coef = torch.linalg.solve(XtWX, XtWy)

        # Get feature importances (exclude bias term)
        importance_scores = coef[:-1]

        return importance_scores


def train_mcal_calibrator(base_model: nn.Module,
                         train_images: torch.Tensor,
                         ablation_rate: float = 0.5,
                         max_steps: int = 5000,
                         batch_size: int = 32,
                         calibrator_type: str = 'mcal_ce',
                         head_type: str = 'scaling',
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train MCal calibration head on ablated training data.

    Args:
        base_model: Uncalibrated model (with softmax)
        train_images: Clean training images
        ablation_rate: Probability of dropping each patch
        max_steps: Maximum training steps
        batch_size: Batch size for training
        device: Device to train on

    Returns:
        mcal_head: Trained MCal calibration head
    """
    print(f"Training MCal calibrator on {device}...")

    # Move model to device
    base_model = base_model.to(device)
    base_model.eval()

    # Initialize MCal head based on calibrator type
    if calibrator_type == 'mcal_ce':
        mcal_head = MCal_CE(num_classes=4, head_type=head_type)
    else:  # mcal_vector
        mcal_head = MCal(num_classes=4)

    # First get clean predictions as targets
    print("Getting clean predictions...")
    clean_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(train_images), batch_size)):
            batch = train_images[i:i+batch_size].to(device)
            probs = base_model(batch)
            clean_probs.append(probs.cpu())

    clean_probs = torch.cat(clean_probs, dim=0)
    clean_preds = clean_probs.argmax(dim=1)  # Use clean predictions as targets

    # Generate ablated training data
    print("Generating ablated training data...")
    n_samples = len(train_images)

    # Load ablated images
    train_ablated, _ = load_mri_ablated_prob(
        split='train',
        p_ablate=ablation_rate,
        n_samples=n_samples
    )

    # Get predictions on ablated images
    print("Getting predictions on ablated images...")
    ablated_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size)):
            batch = train_ablated[i:i+batch_size].to(device)
            probs = base_model(batch)
            ablated_probs.append(probs.cpu())

    ablated_probs = torch.cat(ablated_probs, dim=0)

    # Train MCal head to calibrate ablated predictions to match clean predictions
    print(f"Training {calibrator_type} to map ablated predictions -> clean predictions...")
    if calibrator_type == 'mcal_ce':
        mcal_head.fit(
            ablated_probs,  # Ablated predictions to calibrate
            clean_preds,     # Clean predictions as targets (NOT ground truth labels!)
            max_steps=max_steps
        )
    else:  # mcal_vector
        # MCal vector-scaling uses clean probabilities expectation as target
        mcal_head.fit(
            ablated_probs,  # Ablated predictions to calibrate
            clean_probs=clean_probs,  # Clean probabilities (will use expectation as target)
            max_steps=max_steps,
            verbose=True
        )

    return mcal_head


def compute_sufficiency(model: nn.Module, image: torch.Tensor,
                       top_k_indices: torch.Tensor, patch_masks: torch.Tensor,
                       true_label: int, device: str) -> float:
    """
    Compute sufficiency metric: keeping only top-k features.
    Lower is better (smaller drop in confidence when keeping important features).
    Following Equation (1) from Hase & Bansal (2021).
    """
    # Create mask keeping only top-k patches
    combined_mask = torch.zeros((224, 224))
    for idx in top_k_indices:
        combined_mask += patch_masks[idx]

    # Apply mask to image
    masked_image = image.clone()
    for c in range(3):
        masked_image[c] = masked_image[c] * combined_mask.to(device)

    # Get predictions
    with torch.no_grad():
        pred_original = model(image.unsqueeze(0))[0, true_label].item()
        pred_masked = model(masked_image.unsqueeze(0))[0, true_label].item()

    # Sufficiency: difference in predicted probabilities (Eq. 1 from paper)
    # Lower values are better
    return pred_original - pred_masked


def compute_comprehensiveness(model: nn.Module, image: torch.Tensor,
                             top_k_indices: torch.Tensor, patch_masks: torch.Tensor,
                             true_label: int, device: str) -> float:
    """
    Compute comprehensiveness metric: removing top-k features.
    Higher is better (larger drop when removing important features).
    Following the definition from Section 3 of Hase & Bansal (2021).
    """
    # Create mask removing top-k patches
    combined_mask = torch.ones((224, 224))
    for idx in top_k_indices:
        combined_mask -= patch_masks[idx]
    combined_mask = torch.clamp(combined_mask, 0, 1)

    # Apply mask to image
    masked_image = image.clone()
    for c in range(3):
        masked_image[c] = masked_image[c] * combined_mask.to(device)

    # Get predictions
    with torch.no_grad():
        pred_original = model(image.unsqueeze(0))[0, true_label].item()
        pred_masked = model(masked_image.unsqueeze(0))[0, true_label].item()

    # Comprehensiveness: difference in predicted probabilities
    # Higher values are better (larger drop when removing important features)
    return pred_original - pred_masked


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MCal Explanation Quality Experiment')
    parser.add_argument('--topk', type=int, default=4,
                        help='Number of top features to select from LIME (default: 4)')
    parser.add_argument('--calibrator', type=str, default='mcal_ce',
                        choices=['mcal_ce', 'mcal_vector'],
                        help='Calibrator type: mcal_ce or mcal_vector (default: mcal_ce)')
    parser.add_argument('--head_type', type=str, default='linear',
                        choices=['linear', 'scaling'],
                        help='Type of MCal_CE calibration head (only used with mcal_ce, default: scaling)')
    parser.add_argument('--n_train', type=int, default=1000,
                        help='Number of training samples (default: 1000)')
    parser.add_argument('--n_test', type=int, default=100,
                        help='Number of test samples (default: 100)')
    parser.add_argument('--lime_samples', type=int, default=100,
                        help='Number of samples for LIME (default: 100)')
    parser.add_argument('--ablation_rate', type=float, default=0.5,
                        help='Probability of dropping each patch during MCal training (default: 0.5)')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Maximum training steps for MCal (default: 5000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()


def main():
    """Run the MCal explanation quality experiment."""

    # Parse arguments
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data (only images needed, not labels)
    print(f"Loading MRI data (n_train={args.n_train}, n_test={args.n_test})...")
    train_images, _ = load_mri_clean(split='train', n_samples=args.n_train)
    test_images, test_labels = load_mri_clean(split='test', n_samples=args.n_test)

    # Load base model
    print("Loading base model...")
    base_model = load_mri_model()

    # Create uncalibrated model (base + softmax)
    uncal_model = nn.Sequential(
        base_model,
        nn.Softmax(dim=1)
    )
    uncal_model = uncal_model.to(device)
    uncal_model.eval()

    # Train MCal calibrator (no labels needed)
    if args.calibrator == 'mcal_ce':
        print(f"Training MCal_CE with head_type='{args.head_type}', ablation_rate={args.ablation_rate}...")
    else:
        print(f"Training MCal vector-scaling with ablation_rate={args.ablation_rate}...")

    mcal_head = train_mcal_calibrator(
        base_model=uncal_model,
        train_images=train_images,
        ablation_rate=args.ablation_rate,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        calibrator_type=args.calibrator,
        head_type=args.head_type,
        device=device
    )

    # Create calibrated model (base + softmax + mcal)
    calib_model = nn.Sequential(
        base_model.to(device),
        nn.Softmax(dim=1),
        mcal_head.to(device)
    )
    calib_model.eval()

    # Initialize LIME explainers
    print(f"Initializing LIME explainers (num_samples={args.lime_samples})...")
    uncal_lime = CustomLIME(uncal_model, num_samples=args.lime_samples)
    calib_lime = CustomLIME(calib_model, num_samples=args.lime_samples)

    # Create patch masks for evaluation
    patch_masks = uncal_lime.create_square_patches()

    # Evaluate on test set
    print("Evaluating explanation quality...")
    results = {
        'uncal': {'sufficiency': [], 'comprehensiveness': []},
        'calib': {'sufficiency': [], 'comprehensiveness': []}
    }

    k = args.topk  # Number of top features to consider

    for i in tqdm(range(len(test_images)), desc="Processing test images"):
        image = test_images[i].to(device)
        label = test_labels[i].item()

        # Generate LIME explanations
        uncal_scores = uncal_lime.explain_instance(image, label)
        calib_scores = calib_lime.explain_instance(image, label)

        # Get top-k patches by absolute importance
        uncal_top_k = torch.argsort(torch.abs(uncal_scores))[-k:]
        calib_top_k = torch.argsort(torch.abs(calib_scores))[-k:]

        # Compute metrics for uncalibrated model
        uncal_suff = compute_sufficiency(uncal_model, image, uncal_top_k, patch_masks, label, device)
        uncal_comp = compute_comprehensiveness(uncal_model, image, uncal_top_k, patch_masks, label, device)

        # Compute metrics for calibrated model
        calib_suff = compute_sufficiency(calib_model, image, calib_top_k, patch_masks, label, device)
        calib_comp = compute_comprehensiveness(calib_model, image, calib_top_k, patch_masks, label, device)

        # Store results (ensure float)
        results['uncal']['sufficiency'].append(float(uncal_suff))
        results['uncal']['comprehensiveness'].append(float(uncal_comp))
        results['calib']['sufficiency'].append(float(calib_suff))
        results['calib']['comprehensiveness'].append(float(calib_comp))

    # Convert to tensors for mean/std
    uncal_suff = torch.tensor(results['uncal']['sufficiency'])
    uncal_comp = torch.tensor(results['uncal']['comprehensiveness'])
    calib_suff = torch.tensor(results['calib']['sufficiency'])
    calib_comp = torch.tensor(results['calib']['comprehensiveness'])

    # Calculate values for display
    uncal_suff_mean = float(uncal_suff.mean())
    uncal_suff_std = float(uncal_suff.std())
    uncal_comp_mean = float(uncal_comp.mean())
    uncal_comp_std = float(uncal_comp.std())

    calib_suff_mean = float(calib_suff.mean())
    calib_suff_std = float(calib_suff.std())
    calib_comp_mean = float(calib_comp.mean())
    calib_comp_std = float(calib_comp.std())

    suff_improve = calib_suff_mean - uncal_suff_mean
    comp_improve = calib_comp_mean - uncal_comp_mean

    # Print results with actual values
    print("\n" + "="*75)
    print(f"{'MCAL EXPLANATION QUALITY RESULTS':^75}")
    print("="*75)

    print(f"\nExperiment Configuration:")
    print(f"  Calibrator: {args.calibrator.upper()}")
    if args.calibrator == 'mcal_ce':
        print(f"  Head Type: {args.head_type}")
    print(f"  Test Samples: {len(test_images)}")
    print(f"  Top-k Features: {k}")
    print(f"  LIME Samples: {args.lime_samples}")
    print("-"*75)

    # Print actual metric values
    print("\nMETRIC VALUES:")
    print("-"*75)
    print(f"{'Model':<20} {'Sufficiency':<25} {'Comprehensiveness':<25}")
    print(f"{'':20} {'(lower is better)':<25} {'(higher is better)':<25}")
    print("-"*75)

    print(f"{'Uncalibrated':<20} {uncal_suff_mean:7.4f} ± {uncal_suff_std:6.4f}      "
          f"{uncal_comp_mean:7.4f} ± {uncal_comp_std:6.4f}")

    print(f"{'Calibrated':<20} {calib_suff_mean:7.4f} ± {calib_suff_std:6.4f}      "
          f"{calib_comp_mean:7.4f} ± {calib_comp_std:6.4f}")

    print("-"*75)

    # Show the change
    print(f"{'Change':<20} {suff_improve:+7.4f} {' ':17} {comp_improve:+7.4f}")

    # Interpret the changes
    suff_status = "✓ improved" if suff_improve < 0 else "✗ worse" if suff_improve > 0 else "= no change"
    comp_status = "✓ improved" if comp_improve > 0 else "✗ worse" if comp_improve < 0 else "= no change"

    print(f"{'Result':<20} {suff_status:<25} {comp_status:<25}")
    print("="*75)

    # Show individual sample results
    print("\nSample-level results (first 5):")
    print("-"*75)
    print(f"{'#':<4} {'Uncal Suff':<12} {'Calib Suff':<12} {'Uncal Comp':<12} {'Calib Comp':<12}")
    print("-"*75)

    for i in range(min(5, len(results['uncal']['sufficiency']))):
        print(f"{i+1:<4} {results['uncal']['sufficiency'][i]:<12.4f} "
              f"{results['calib']['sufficiency'][i]:<12.4f} "
              f"{results['uncal']['comprehensiveness'][i]:<12.4f} "
              f"{results['calib']['comprehensiveness'][i]:<12.4f}")

    print("="*75)

    # Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, 'mcal_explanation_quality_mri.json')

    save_results = {
        'uncalibrated': {
            'sufficiency_mean': uncal_suff_mean,
            'sufficiency_std': uncal_suff_std,
            'comprehensiveness_mean': uncal_comp_mean,
            'comprehensiveness_std': uncal_comp_std,
            'sufficiency_values': [float(x) for x in results['uncal']['sufficiency']],
            'comprehensiveness_values': [float(x) for x in results['uncal']['comprehensiveness']]
        },
        'calibrated': {
            'sufficiency_mean': calib_suff_mean,
            'sufficiency_std': calib_suff_std,
            'comprehensiveness_mean': calib_comp_mean,
            'comprehensiveness_std': calib_comp_std,
            'sufficiency_values': [float(x) for x in results['calib']['sufficiency']],
            'comprehensiveness_values': [float(x) for x in results['calib']['comprehensiveness']]
        },
        'improvement': {
            'sufficiency': suff_improve,  # Negative is better (lower sufficiency is better)
            'comprehensiveness': comp_improve  # Positive is better (higher comprehensiveness is better)
        },
        'config': {
            'n_train': len(train_images),
            'n_test': len(test_images),
            'k': k,
            'lime_samples': args.lime_samples,
            'mcal_steps': args.max_steps,
            'ablation_rate': args.ablation_rate,
            'batch_size': args.batch_size,
            'calibrator': args.calibrator,
            'head_type': args.head_type if args.calibrator == 'mcal_ce' else 'N/A',
            'seed': args.seed
        }
    }

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()