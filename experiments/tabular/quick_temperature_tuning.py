#!/usr/bin/env python3
"""
Quick temperature scaling hyperparameter analysis script.

This script analyzes the effect of different hyperparameters on temperature scaling
performance using your existing benchmark infrastructure.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from calibrators.temperature import TemperatureScaling


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def test_learning_rates(train_probs, train_labels, val_probs, val_labels, device):
    """Test different learning rates."""
    learning_rates = [0.001, 0.01, 0.1, 1.0, 10.0]
    results = []

    print("Testing learning rates...")
    for lr in tqdm(learning_rates):
        model = TemperatureScaling(num_classes=train_probs.shape[1])
        model.to(device)

        try:
            stats = model.fit(
                ablated_probs=train_probs,
                labels=train_labels,
                max_steps=1000,
                lr=lr,
                verbose=False
            )

            with torch.no_grad():
                calibrated_probs = model.forward(val_probs)
                ece = compute_ece(calibrated_probs, val_labels)

            results.append({
                'lr': lr,
                'ece': ece,
                'final_temperature': model.temperature.item(),
                'converged_steps': len(stats['loss']),
                'final_loss': stats['loss'][-1] if stats['loss'] else float('inf')
            })

        except Exception as e:
            results.append({
                'lr': lr,
                'ece': float('inf'),
                'error': str(e)
            })

    return results


def test_max_steps(train_probs, train_labels, val_probs, val_labels, device):
    """Test different maximum steps."""
    max_steps_list = [50, 100, 500, 1000, 2000, 5000]
    results = []

    print("Testing max steps...")
    for max_steps in tqdm(max_steps_list):
        model = TemperatureScaling(num_classes=train_probs.shape[1])
        model.to(device)

        try:
            stats = model.fit(
                ablated_probs=train_probs,
                labels=train_labels,
                max_steps=max_steps,
                lr=0.01,
                verbose=False
            )

            with torch.no_grad():
                calibrated_probs = model.forward(val_probs)
                ece = compute_ece(calibrated_probs, val_labels)

            results.append({
                'max_steps': max_steps,
                'ece': ece,
                'final_temperature': model.temperature.item(),
                'actual_steps': len(stats['loss']),
                'final_loss': stats['loss'][-1] if stats['loss'] else float('inf')
            })

        except Exception as e:
            results.append({
                'max_steps': max_steps,
                'ece': float('inf'),
                'error': str(e)
            })

    return results


def test_temperature_initialization(train_probs, train_labels, val_probs, val_labels, device):
    """Test different temperature initializations."""
    temp_inits = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []

    print("Testing temperature initializations...")
    for temp_init in tqdm(temp_inits):
        model = TemperatureScaling(num_classes=train_probs.shape[1])
        model.to(device)

        # Set initial temperature
        with torch.no_grad():
            model.temperature.fill_(temp_init)

        try:
            stats = model.fit(
                ablated_probs=train_probs,
                labels=train_labels,
                max_steps=1000,
                lr=0.01,
                verbose=False
            )

            with torch.no_grad():
                calibrated_probs = model.forward(val_probs)
                ece = compute_ece(calibrated_probs, val_labels)

            results.append({
                'temp_init': temp_init,
                'ece': ece,
                'final_temperature': model.temperature.item(),
                'convergence_steps': len(stats['loss']),
                'final_loss': stats['loss'][-1] if stats['loss'] else float('inf')
            })

        except Exception as e:
            results.append({
                'temp_init': temp_init,
                'ece': float('inf'),
                'error': str(e)
            })

    return results


def plot_results(results, param_name, save_path):
    """Plot hyperparameter tuning results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Extract valid results
    valid_results = [r for r in results if 'error' not in r and r['ece'] != float('inf')]

    if not valid_results:
        print(f"No valid results for {param_name}")
        return

    params = [r[param_name] for r in valid_results]
    eces = [r['ece'] for r in valid_results]
    temps = [r['final_temperature'] for r in valid_results]
    losses = [r['final_loss'] for r in valid_results]

    # Plot ECE vs parameter
    ax1.plot(params, eces, 'bo-')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('ECE')
    ax1.set_title(f'ECE vs {param_name}')
    ax1.grid(True)

    # Plot final temperature vs parameter
    ax2.plot(params, temps, 'ro-')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Final Temperature')
    ax2.set_title(f'Final Temperature vs {param_name}')
    ax2.grid(True)

    # Plot final loss vs parameter
    ax3.plot(params, losses, 'go-')
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Final Loss')
    ax3.set_title(f'Final Loss vs {param_name}')
    ax3.grid(True)

    # Create a summary text
    best_idx = np.argmin(eces)
    best_result = valid_results[best_idx]
    summary_text = f"Best {param_name}: {best_result[param_name]}\n"
    summary_text += f"Best ECE: {best_result['ece']:.6f}\n"
    summary_text += f"Final Temp: {best_result['final_temperature']:.4f}\n"
    summary_text += f"Final Loss: {best_result['final_loss']:.6f}"

    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Best Configuration')
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def main():
    """Main function to run hyperparameter analysis."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load some sample data from breast cancer dataset
    sys.path.insert(0, str(Path(__file__).parent))
    from breast_cancer_data_setup import load_breast_cancer_data

    print("Loading breast cancer data for tuning...")
    predictions_tensor, labels_tensor = load_breast_cancer_data(
        model_type="vanilla",
        n_samples=500,  # Small sample for quick tuning
        n_fractions=1,
        fill_value="mean"
    )

    # Split data into train/val (use fraction 0 which is unablated)
    n_samples = predictions_tensor.shape[1]
    n_train = int(0.8 * n_samples)

    # Extract data - predictions_tensor shape is (n_fractions, n_samples, n_classes)
    all_probs = predictions_tensor[0]  # Use fraction 0 (unablated data)
    all_labels = labels_tensor

    # Split into train/val
    train_probs = all_probs[:n_train].to(device)
    train_labels = all_labels[:n_train].to(device)
    val_probs = all_probs[n_train:].to(device)
    val_labels = all_labels[n_train:].to(device)

    print(f"Data loaded: {train_probs.shape[0]} train, {val_probs.shape[0]} val")

    # Create output directory
    output_dir = Path("./temperature_tuning_results")
    output_dir.mkdir(exist_ok=True)

    # Test different hyperparameters
    all_results = {}

    # Test learning rates
    lr_results = test_learning_rates(train_probs, train_labels, val_probs, val_labels, device)
    all_results['learning_rates'] = lr_results
    plot_results(lr_results, 'lr', output_dir / 'learning_rates.png')

    # Test max steps
    steps_results = test_max_steps(train_probs, train_labels, val_probs, val_labels, device)
    all_results['max_steps'] = steps_results
    plot_results(steps_results, 'max_steps', output_dir / 'max_steps.png')

    # Test temperature initialization
    temp_results = test_temperature_initialization(train_probs, train_labels, val_probs, val_labels, device)
    all_results['temp_init'] = temp_results
    plot_results(temp_results, 'temp_init', output_dir / 'temp_init.png')

    # Save all results
    output_file = output_dir / 'temperature_tuning_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {output_file}")

    # Print best configurations
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS")
    print("="*60)

    for param_type, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r and r['ece'] != float('inf')]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['ece'])
            print(f"\nBest {param_type}:")
            for key, value in best_result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()