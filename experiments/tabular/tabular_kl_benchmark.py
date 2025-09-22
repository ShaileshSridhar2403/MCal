#!/usr/bin/env python3
"""
Tabular KL Divergence Benchmark - MCal Implementation

Following the exact pattern of vision and language KL benchmarks but for tabular data
with XGBoost models and PhysioNet dataset.

Self-contained implementation following MCal conventions.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from tabulate import tabulate

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))
sys.path.insert(0, str(mcal_root / "configs"))
sys.path.insert(0, str(mcal_root / "src"))

# Import MCal utilities
from src.utils.optimization import get_expectation, make_one_hot, kl_divergence

# Import calibrator modules
from src.calibrators.mcal import MCal
from src.calibrators.mcal_ce import MCal_CE
from src.calibrators.platt import PlattCalibrator
from src.calibrators.temperature import TemperatureScaling

# Import transform modules
from src.transforms.lambda_transforms import ExpectationLambdaTransform, OptimizedLambdaTransform
from src.transforms.logits import LogitsSharpTransform

# Import our tabular utilities (ensure local imports take precedence)
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from physionet_data_setup import load_physionet_data
from xgboost_utils import MCALXGBoostPredictor
from tabular_utils import (
    aggregate_results,
    build_kl_comparison_table,
    save_results,
    convert_to_json_serializable,
    plot_kl_divergence
)

# Add XAI_Benchmark optimization for LogitsSharp (after our imports)
try:
    xai_root = mcal_root.parent / "XAI-Benchmark"
    sys.path.append(str(xai_root))
    sys.path.append(str(xai_root / "optimization"))
    from optimization.optimization_utils import process_outputs
except ImportError:
    print("Warning: Could not import LogitsSharp from XAI_Benchmark")
    process_outputs = None


def calculate_kl_metrics(outputs, labels=None, device=None):
    """Calculate KL divergence metrics and accuracy for outputs - IDENTICAL to vision benchmarks."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_fractions, n_samples, n_outputs = outputs.shape

    # Results storage
    kl_values_argmax = []
    kl_values_prob = []
    accuracy_values = []

    for fraction in range(n_fractions):
        fraction_preds = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)

        # Get expectations
        one_hot_expectation, prob_expectation = get_expectation(fraction_preds, device)

        # Clean distribution for comparison (fraction 0 = no ablation)
        clean_dist = torch.tensor(outputs[0], dtype=torch.float32, device=device).mean(dim=0)

        # Calculate KL divergences
        kl_argmax = kl_divergence(one_hot_expectation, clean_dist).item()
        kl_prob = kl_divergence(prob_expectation, clean_dist).item()

        kl_values_argmax.append(kl_argmax)
        kl_values_prob.append(kl_prob)

        # Calculate accuracy if labels are provided
        if labels is not None:
            predicted_labels = np.argmax(outputs[fraction], axis=1)
            if isinstance(labels, torch.Tensor):
                true_labels = labels.cpu().numpy()
            else:
                true_labels = np.array(labels)

            predicted_labels = np.array(predicted_labels)
            true_labels = np.array(true_labels)
            accuracy = float(np.mean(predicted_labels == true_labels))
            accuracy_values.append(accuracy)

            print(f"Fraction {fraction}/{n_fractions} - KL Argmax: {kl_argmax:.6f}, KL Prob: {kl_prob:.6f}, Accuracy: {accuracy:.4f}")
        else:
            print(f"Fraction {fraction}/{n_fractions} - KL Argmax: {kl_argmax:.6f}, KL Prob: {kl_prob:.6f}")

    # Calculate averages
    avg_kl_argmax = np.mean(kl_values_argmax)
    avg_kl_prob = np.mean(kl_values_prob)

    result = {
        'average_kl_argmax': avg_kl_argmax,
        'average_kl_prob': avg_kl_prob,
        'kl_values_argmax': kl_values_argmax,
        'kl_values_prob': kl_values_prob
    }

    if accuracy_values:
        avg_accuracy = np.mean(accuracy_values)
        result['accuracy_values'] = accuracy_values
        result['average_accuracy'] = avg_accuracy

    return result


def apply_transform(outputs, labels, method, device=None, **kwargs):
    """Apply a transformation method to outputs - IDENTICAL to vision benchmarks."""
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

    else:
        raise ValueError(f"Unknown transform method: {method}")


def apply_mcal_calibrator(outputs, device, kappa=4.0, max_steps=10000, **kwargs):
    """Apply MCal calibrator using uniform target distribution."""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)

    # Create uniform target distribution
    uniform_target = torch.ones(n_classes, device=device) / n_classes

    # Train one MCal calibrator per fraction
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
                           head_type="linear", experiment_id="physionet_experiment", **kwargs):
    """Apply MCal_CE calibrator using cross-entropy loss."""
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

        calibrated_probs = calibrator.forward(outputs_tensor[fraction])
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    # Combine results
    print(f"\n=== Combining MCal_CE results for experiment: {experiment_id} ===")
    combined_file = MCal_CE.combine_fraction_results(experiment_id, cleanup_temp_files=True)
    if combined_file:
        print(f"All MCal_CE results combined and saved to: {combined_file}")

    return transformed_outputs


def apply_platt_calibrator(outputs, labels, device, max_steps=1000, **kwargs):
    """Apply Platt scaling calibrator."""
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
    """Apply temperature scaling calibrator."""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)

    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    for fraction in tqdm(range(n_fractions), desc="Applying Temperature calibrator"):
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)

        calibrator = TemperatureScaling(num_classes=n_classes)
        calibrator.to(device)
        calibrator.fit(
            ablated_probs=ablated_probs,
            labels=labels_tensor,
            max_steps=max_steps,
            verbose=False
        )

        calibrated_probs = calibrator.forward(ablated_probs)
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    return transformed_outputs


def apply_logits_sharp_transform(outputs, device, num_epochs=1000, **kwargs):
    """Apply LogitsSharp transform from XAI_Benchmark."""
    if process_outputs is None:
        print("⚠️  LogitsSharp transform not available (XAI_Benchmark not found)")
        return outputs

    # Save outputs temporarily for LogitsSharp processing
    temp_path = "/tmp/tabular_outputs_for_logits_sharp.npy"
    np.save(temp_path, outputs)

    try:
        # Apply LogitsSharp transform using XAI_Benchmark's process_outputs
        result = process_outputs(
            temp_path,
            output_type='optimized',
            transform_method='logits_sharp',
            device=str(device),
            num_epochs=num_epochs,
            overwrite=True
        )

        # Extract transformed outputs
        if 'kl_results_transformed' in result and 'transformed_outputs' in result:
            transformed_outputs = result['transformed_outputs']
            print(f"✓ LogitsSharp transform applied successfully")
            return transformed_outputs
        else:
            print("⚠️  LogitsSharp transform failed, returning original outputs")
            return outputs

    except Exception as e:
        print(f"⚠️  LogitsSharp transform error: {str(e)}")
        return outputs
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Stub implementations for other transforms
def apply_expectation_prob_transform(outputs, device):
    """Stub - apply expectation prob transform."""
    print("⚠️  Expectation prob transform not implemented yet")
    return outputs


def apply_expectation_onehot_transform(outputs, device):
    """Stub - apply expectation onehot transform."""
    print("⚠️  Expectation onehot transform not implemented yet")
    return outputs


def apply_optimized_lambda_transform(outputs, device, **kwargs):
    """Stub - apply optimized lambda transform."""
    print("⚠️  Optimized lambda transform not implemented yet")
    return outputs


def process_physionet_dataset(methods=None, device="cuda", save_dir="./results", n_runs=3,
                            n_samples=1000, n_fractions=10, missingness_range="0-30",
                            imputation_strategy="mean"):
    """Process PhysioNet dataset and generate KL benchmarks - IDENTICAL STRUCTURE to vision."""

    if methods is None:
        methods = ['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp', 'retrain']

    device = torch.device(device)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)

    print("="*60)
    print("PhysioNet Tabular KL Divergence Benchmark")
    print("="*60)
    print(f"Methods: {methods}")
    print(f"Runs: {n_runs}")
    print(f"Samples per run: {n_samples}")
    print(f"Fractions: {n_fractions}")
    print(f"Missingness range: {missingness_range}")
    print(f"Imputation strategy: {imputation_strategy}")
    print(f"Device: {device}")

    # Initialize results storage
    all_results = {method: [] for method in methods}

    # Run multiple experiments
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Load data for each method (some methods need different models)
        method_predictions = {}

        # Load standard data for most methods
        if any(method in ['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp'] for method in methods):
            predictions, labels = load_physionet_data(
                model_type=imputation_strategy,
                n_samples=n_samples,
                n_fractions=n_fractions,
                missingness_range=missingness_range,
                balanced=True
            )
            print(f"Loaded standard data - Predictions: {predictions.shape}, Labels: {labels.shape}")

            # Store for standard methods
            for method in ['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp']:
                if method in methods:
                    method_predictions[method] = predictions

        # Load retrain data if needed
        if 'retrain' in methods:
            retrain_predictions, retrain_labels = load_physionet_data(
                model_type="retrain",
                n_samples=n_samples,
                n_fractions=n_fractions,
                missingness_range=missingness_range,
                balanced=True
            )
            print(f"Loaded retrain data - Predictions: {retrain_predictions.shape}, Labels: {retrain_labels.shape}")
            method_predictions['retrain'] = retrain_predictions
            labels = retrain_labels  # Use retrain labels (should be same as standard)

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
                    method_kwargs['experiment_id'] = f'physionet_run_{run}'
                elif method == 'logits_sharp':
                    method_kwargs['num_epochs'] = 1000

                transformed_predictions = apply_transform(
                    method_preds.numpy(), labels.numpy(), method, device, **method_kwargs
                )

            # Calculate KL metrics with accuracy
            kl_results = calculate_kl_metrics(transformed_predictions, labels.numpy(), device)
            all_results[method].append(kl_results)

            print(f"  KL (prob): {kl_results['average_kl_prob']:.6f}")
            print(f"  KL (argmax): {kl_results['average_kl_argmax']:.6f}")
            if 'average_accuracy' in kl_results:
                print(f"  Average accuracy: {kl_results['average_accuracy']:.4f}")

    # Aggregate results
    print("\nAggregating results across all runs...")
    aggregated_results = aggregate_results(all_results)

    # Save results
    json_path, table_path, plot_path = save_results(
        aggregated_results, save_dir, "PhysioNet", n_runs
    )

    return aggregated_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="PhysioNet Tabular KL Divergence Benchmark")
    parser.add_argument("--methods", nargs='+',
                       default=['baseline', 'mcal', 'platt', 'temperature', 'logits_sharp', 'retrain'],
                       help="Methods to include in benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per run")
    parser.add_argument("--fractions", type=int, default=10, help="Number of fractions")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="./results", help="Save directory")
    parser.add_argument("--missingness_range", type=str, default="0-30",
                       choices=['0-30', '30-100', 'full'],
                       help="PhysioNet missingness range")
    parser.add_argument("--imputation_strategy", type=str, default="mean",
                       choices=['mean', 'zero', 'xgboost_native'],
                       help="Model imputation strategy")

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {device}")

    # Run benchmark
    aggregated_results = process_physionet_dataset(
        methods=args.methods,
        device=device,
        save_dir=args.save_dir,
        n_runs=args.runs,
        n_samples=args.samples,
        n_fractions=args.fractions,
        missingness_range=args.missingness_range,
        imputation_strategy=args.imputation_strategy
    )

    print("\nTabular benchmark completed! 🎉")

    # Print final summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for method, results in aggregated_results.items():
        if 'kl_transformed_mean_prob' in results:
            print(f"{method.upper()}:")
            print(f"  KL Prob: {results['kl_transformed_mean_prob']:.2e} ± {results['kl_transformed_std_prob']:.2e}")
            print(f"  KL Argmax: {results['kl_transformed_mean_onehot']:.2e} ± {results['kl_transformed_std_onehot']:.2e}")


if __name__ == "__main__":
    main()