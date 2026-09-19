#!/usr/bin/env python3
"""
Clean CTG KL Divergence Benchmark

This benchmark file only handles:
1. Loading data via ctg_data_setup.py
2. Applying transforms (calibration methods)
3. Calculating KL metrics
4. Saving results

All data generation logic is in ctg_data_setup.py
(Following the PhysioNet benchmark pattern exactly)
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json
from tabulate import tabulate

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent

# Import MCal components
from mcal.utils.optimization import make_one_hot
from mcal.calibrators.mcal import MCal
from mcal.calibrators.mcal_ce import MCal_CE
from experiments.mcal_ce_results import save_fit_summary, combine_fraction_results

# Per-fraction and combined MCal_CE summaries are written here
MCAL_CE_RESULTS_DIR = Path(__file__).parent / "results"
from mcal.calibrators.platt import PlattCalibrator
from mcal.calibrators.temperature import TemperatureScaling

# Import data setup (from current directory)
from experiments.tabular.ctg_data_setup import load_ctg_data

# Import result utilities (from current directory)
try:
    from experiments.tabular.tabular_utils import (
        aggregate_results,
        build_kl_comparison_table,
        save_results,
        convert_to_json_serializable,
        plot_kl_divergence
    )
except ImportError:
    print("Warning: tabular_utils not found - using fallback functions")

    # Fallback functions
    def aggregate_results(results):
        return results

    def build_kl_comparison_table(results):
        return str(results)

    def save_results(results, save_dir, dataset_name):
        json_path = f"{save_dir}/results.json"
        with open(json_path, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        return json_path, None, None

    def convert_to_json_serializable(obj):
        return obj

    def plot_kl_divergence(results, save_path):
        pass

# MCal optimization utilities
from mcal.utils.optimization import kl_divergence, get_expectation


def calculate_kl_metrics(outputs, labels=None, device=None):
    """
    Calculate KL divergence metrics using expectation approach.
    (Identical to PhysioNet version)

    Args:
        outputs (np.ndarray): Model predictions of shape (n_fractions, n_samples, n_classes)
        labels (np.ndarray): True labels (optional, for additional metrics)
        device: PyTorch device

    Returns:
        dict: KL metrics including average_kl_prob, average_kl_argmax, average_accuracy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_fractions, n_samples, n_outputs = outputs.shape

    # Convert to torch tensor if needed
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs, dtype=torch.float32, device=device)

    # Calculate expectation for each fraction
    expectations_prob = []
    expectations_argmax = []
    accuracies = []

    for fraction in range(n_fractions):
        probs = outputs[fraction]

        # Ensure probs is on the correct device
        probs = probs.to(device)

        # Expectation of probabilities
        _, exp_prob = get_expectation(probs)
        expectations_prob.append(exp_prob)

        # Expectation of argmax (one-hot)
        one_hot = make_one_hot(probs)
        exp_argmax, _ = get_expectation(one_hot)
        expectations_argmax.append(exp_argmax)

        # Calculate accuracy if labels provided
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            else:
                labels = labels.to(device)
            pred_labels = probs.argmax(dim=-1)

            accuracy = (pred_labels == labels).float().mean().item()
            accuracies.append(accuracy)

    # Stack expectations
    expectations_prob = torch.stack(expectations_prob)  # (n_fractions, n_classes)
    expectations_argmax = torch.stack(expectations_argmax)  # (n_fractions, n_classes)

    # Calculate KL divergences (mean over fractions)
    kl_prob = kl_divergence(expectations_prob[0], expectations_prob.mean(dim=0))
    kl_argmax = kl_divergence(expectations_argmax[0], expectations_argmax.mean(dim=0))

    results = {
        'average_kl_prob': kl_prob.item(),
        'average_kl_argmax': kl_argmax.item(),
    }

    if labels is not None and accuracies:
        results['average_accuracy'] = np.mean(accuracies)

    return results


def apply_transform(outputs, labels, method, device=None, **kwargs):
    """Apply a transformation method to outputs. (Identical to PhysioNet version)"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Applying {method} transform...")

    if method == 'baseline' or method == 'retrain':
        return outputs

    elif method == 'mcal':
        return apply_mcal_calibrator(outputs, device, **kwargs)

    elif method == 'mcal_ce':
        return apply_mcal_ce_calibrator(outputs, labels, device, **kwargs)

    elif method == 'platt':
        return apply_platt_calibrator(outputs, labels, device, **kwargs)

    elif method == 'temperature':
        return apply_temperature_calibrator(outputs, labels, device, **kwargs)

    elif method == 'logits_sharp':
        return apply_logits_sharp_transform(outputs, device, **kwargs)

    elif method == 'expectation_prob':
        return apply_expectation_prob_transform(outputs, device)

    elif method == 'expectation_onehot':
        return apply_expectation_onehot_transform(outputs, device)

    elif method == 'optimized_lambda':
        return apply_optimized_lambda_transform(outputs, device, **kwargs)

    elif method == 'replace':
        return outputs  # No transformation - zero-fill applied during data generation

    elif method == 'archmod':
        return apply_archmod_transform(outputs, device, **kwargs)

    else:
        print(f"Unknown method: {method}")
        return outputs


def apply_mcal_calibrator(outputs, device, kappa=4.0, max_steps=10000, **kwargs):
    """Apply MCal calibrator. (Adapted for 3-class CTG)"""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)

    # Create uniform target distribution for 3 classes
    uniform_target = torch.full((n_classes,), 1.0 / n_classes, device=device)

    # Train calibrators for each fraction
    calibrators = []
    for fraction in tqdm(range(n_fractions), desc="Training MCal calibrators"):
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)

        calibrator = MCal(num_classes=n_classes, target_distribution=uniform_target)
        calibrator.to(device)
        calibrator.fit(
            ablated_probs=ablated_probs,
            target_distribution=uniform_target,
            kappa=kappa,
            max_steps=max_steps,
            lr=1e-1,
            verbose=False
        )
        calibrators.append(calibrator)

    # Apply calibration
    for fraction in tqdm(range(n_fractions), desc="Applying MCal calibration"):
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        calibrated_probs = calibrators[fraction].forward(ablated_probs)
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    return transformed_outputs


def apply_mcal_ce_calibrator(outputs_tensor, target_labels, device, max_steps=5000,
                           head_type="linear", experiment_id="ctg_experiment", **kwargs):
    """Apply MCal_CE calibrator using cross-entropy loss. (Identical to PhysioNet version)"""
    # Convert numpy arrays to torch tensors if needed
    if not isinstance(outputs_tensor, torch.Tensor):
        outputs_tensor = torch.tensor(outputs_tensor, dtype=torch.float32, device=device)
    if not isinstance(target_labels, torch.Tensor):
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

    # Use clean predictions (fraction 0) as target labels
    target_labels = outputs_tensor[0].argmax(dim=-1)

    n_fractions, n_samples, n_classes = outputs_tensor.shape
    transformed_outputs = np.zeros_like(outputs_tensor.cpu().numpy())

    for fraction in tqdm(range(n_fractions), desc="Applying MCal_CE calibrator"):
        calibrator = MCal_CE(num_classes=n_classes, head_type="mlp")
        calibrator.to(device)
        calibrator.fit(
            ablated_probs=outputs_tensor[fraction],
            target_labels=target_labels,
            max_steps=max_steps,
            lr=1e-2,
            verbose=False,
            fraction=fraction,
            experiment_id=experiment_id
        )
        save_fit_summary(calibrator, MCAL_CE_RESULTS_DIR)

        calibrated_probs = calibrator.forward(outputs_tensor[fraction])
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    # Combine results
    print(f"\n=== Combining MCal_CE results for experiment: {experiment_id} ===")
    combined_file = combine_fraction_results(MCAL_CE_RESULTS_DIR, experiment_id, cleanup_temp_files=True)
    if combined_file:
        print(f"All MCal_CE results combined and saved to: {combined_file}")

    return transformed_outputs


def apply_platt_calibrator(outputs, labels, device, max_steps=1000, **kwargs):
    """Apply Platt scaling calibrator. (Identical to PhysioNet version)"""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)

    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    # Fit on fraction 0 (unablated)
    unablated_probs = torch.tensor(outputs[0], dtype=torch.float32, device=device)
    calibrator = PlattCalibrator(num_classes=n_classes)
    calibrator.to(device)
    calibrator.fit(
        ablated_probs=unablated_probs,
        labels=labels_tensor,
        max_steps=max_steps,
        verbose=False
    )

    # Apply to all fractions
    for fraction in tqdm(range(n_fractions), desc="Applying Platt calibrator"):
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        calibrated_probs = calibrator.forward(ablated_probs)
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    return transformed_outputs


def apply_temperature_calibrator(outputs, labels, device, max_steps=1000, **kwargs):
    """Apply temperature scaling calibrator. (Identical to PhysioNet version)"""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)

    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    # Fit on fraction 0 (unablated)
    unablated_probs = torch.tensor(outputs[0], dtype=torch.float32, device=device)
    calibrator = TemperatureScaling(num_classes=n_classes)
    calibrator.to(device)
    calibrator.fit(
        ablated_probs=unablated_probs,
        labels=labels_tensor,
        max_steps=max_steps,
        verbose=False
    )

    # Apply to all fractions
    for fraction in tqdm(range(n_fractions), desc="Applying temperature calibrator"):
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        calibrated_probs = calibrator.forward(ablated_probs)
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    return transformed_outputs


def apply_logits_sharp_transform(outputs, device, num_epochs=1000, **kwargs):
    """Apply LogitsSharp transform. (Identical to PhysioNet version)"""
    print("⚠️  LogitsSharp transform not yet implemented for tabular data")
    return outputs


def apply_expectation_prob_transform(outputs, device):
    """Apply expectation probability transform. (Identical to PhysioNet version)"""
    print("⚠️  Expectation prob transform not implemented yet")
    return outputs


def apply_expectation_onehot_transform(outputs, device):
    """Apply expectation one-hot transform. (Identical to PhysioNet version)"""
    print("⚠️  Expectation one-hot transform not implemented yet")
    return outputs


def apply_optimized_lambda_transform(outputs, device, **kwargs):
    """Stub - apply optimized lambda transform. (Identical to PhysioNet version)"""
    print("⚠️  Optimized lambda transform not implemented yet")
    return outputs


def apply_archmod_transform(outputs, device, **kwargs):
    """Apply architecture modification transform using custom missing value (-10) handling."""
    return outputs  # No transformation - custom missing value handling applied during data generation


def process_ctg_dataset(methods=None, device="cuda", save_dir="./results", n_runs=3,
                       n_samples=1000, n_fractions=10):
    """Process CTG dataset and generate KL benchmarks. (Adapted from PhysioNet pattern)"""

    if methods is None:
        methods = ['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp', 'retrain', 'replace', 'archmod']

    device = torch.device(device)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)

    print("="*60)
    print("CTG Tabular KL Divergence Benchmark")
    print("="*60)
    print(f"Methods: {methods}")
    print(f"Runs: {n_runs}")
    print(f"Samples per run: {n_samples}")
    print(f"Fractions: {n_fractions}")
    print(f"Fill value: mean")
    print(f"Device: {device}")

    # Initialize results storage
    all_results = {method: [] for method in methods}

    # Run multiple experiments
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Load data based on method requirements
        method_predictions = {}

        # Standard methods use vanilla model with mean imputation
        if any(method in ['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp'] for method in methods):
            predictions, labels = load_ctg_data(
                model_type="vanilla",  # Clean, simple vanilla model
                fill_value="mean",     # Mean imputation for missing values
                n_samples=n_samples,
                n_fractions=n_fractions
            )
            print(f"Loaded vanilla data - Predictions: {predictions.shape}, Labels: {labels.shape}")

            # Store for all standard methods
            for method in ['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp']:
                if method in methods:
                    method_predictions[method] = predictions

        # Retrain method uses retrained model
        if 'retrain' in methods:
            retrain_predictions, retrain_labels = load_ctg_data(
                model_type="retrained",  # Uses 50% binomial missingness training
                fill_value="mean",       # Mean imputation for missing values
                n_samples=n_samples,
                n_fractions=n_fractions
            )
            method_predictions['retrain'] = retrain_predictions
            # Use retrain labels if no other labels available
            if 'labels' not in locals():
                labels = retrain_labels

        # Replace method uses vanilla model with zero-fill
        if 'replace' in methods:
            replace_predictions, replace_labels = load_ctg_data(
                model_type="vanilla",    # Same model as standard methods
                fill_value="zero",       # BUT zero-fill instead of mean
                n_samples=n_samples,
                n_fractions=n_fractions
            )
            method_predictions['replace'] = replace_predictions
            print(f"Loaded replace data - Predictions: {replace_predictions.shape}, Labels: {replace_labels.shape}")
            # Use replace labels if no other labels available
            if 'labels' not in locals():
                labels = replace_labels

        # ArchMod method uses vanilla model with -10 fill and custom missing parameter
        if 'archmod' in methods:
            archmod_predictions, archmod_labels = load_ctg_data(
                model_type="vanilla",    # Same model architecture
                fill_value="-10",        # Fill missing with -10
                n_samples=n_samples,
                n_fractions=n_fractions,
                missing_value=-10        # Tell XGBoost that -10 = missing
            )
            method_predictions['archmod'] = archmod_predictions
            print(f"Loaded archmod data - Predictions: {archmod_predictions.shape}, Labels: {archmod_labels.shape}")
            # Use archmod labels if no other labels available
            if 'labels' not in locals():
                labels = archmod_labels

        # Process each method
        for method in methods:
            print(f"\nProcessing method: {method}")

            # Get predictions for this method
            method_preds = method_predictions[method]

            # Apply transformation
            if method == 'baseline' or method == 'retrain':
                transformed_predictions = method_preds
            else:
                # Configure method-specific parameters
                method_kwargs = {}
                if method in ['mcal', 'mcal_ce', 'platt', 'temperature']:
                    method_kwargs['max_steps'] = 1000
                if method == 'mcal':
                    method_kwargs['kappa'] = 10.0
                elif method == 'mcal_ce':
                    method_kwargs['max_steps'] = 5000
                    method_kwargs['head_type'] = 'linear'
                    method_kwargs['experiment_id'] = f'ctg_run_{run}'
                elif method == 'logits_sharp':
                    method_kwargs['num_epochs'] = 1000

                transformed_predictions = apply_transform(
                    method_preds.numpy(), labels.numpy(), method, device, **method_kwargs
                )

            # Calculate KL metrics with accuracy
            kl_results = calculate_kl_metrics(transformed_predictions, labels, device)
            all_results[method].append(kl_results)

            print(f"  KL (prob): {kl_results['average_kl_prob']:.6f}")
            print(f"  KL (argmax): {kl_results['average_kl_argmax']:.6f}")
            if 'average_accuracy' in kl_results:
                print(f"  Accuracy: {kl_results['average_accuracy']:.4f}")

    # Aggregate and save results
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    aggregated_results = aggregate_results(all_results)
    json_path, table_path, plot_path = save_results(aggregated_results, save_dir, "ctg", n_runs)

    # Print summary table
    print("\nFinal Results Summary:")
    table = build_kl_comparison_table(aggregated_results)
    print(table)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {save_dir}")
    print(f"JSON: {json_path}")
    print(f"Table: {table_path}")
    print(f"Plot: {plot_path}")

    return aggregated_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="CTG Tabular KL Divergence Benchmark")
    parser.add_argument("--methods", nargs="+",
                       choices=['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp', 'retrain', 'replace', 'archmod'],
                       default=['baseline', 'mcal_ce', 'retrain'],
                       help="Methods to benchmark")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--save_dir", default=str(Path(__file__).parent / "results"), help="Directory to save results")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples per run")
    parser.add_argument("--n_fractions", type=int, default=10, help="Number of ablation fractions")

    args = parser.parse_args()

    # Run benchmark
    results = process_ctg_dataset(
        methods=args.methods,
        device=args.device,
        save_dir=args.save_dir,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        n_fractions=args.n_fractions
    )

    print("Benchmark completed successfully!")
    return results


if __name__ == "__main__":
    main()