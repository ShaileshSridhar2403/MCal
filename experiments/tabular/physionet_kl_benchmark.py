#!/usr/bin/env python3
"""
Clean PhysioNet KL Divergence Benchmark

This benchmark file only handles:
1. Loading data via physionet_data_setup.py
2. Applying transforms (calibration methods)
3. Calculating KL metrics
4. Saving results

All data generation logic is in physionet_data_setup_simple.py
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent

# Import MCal components
from mcal.utils.optimization import make_one_hot
from mcal.calibrators.mcal import MCal
from mcal.calibrators.mcal_ce import MCal_CE
from experiments.mcal_ce_results import save_fit_summary, combine_fraction_results

from mcal.calibrators.platt import PlattCalibrator
from mcal.calibrators.temperature import TemperatureScaling

# Import data setup (from current directory)

from experiments.tabular.physionet_data_setup import load_physionet_data




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
    def aggregate_results(all_results):
        """Aggregate results across multiple runs with fractionwise KL and accuracy calculation."""
        aggregated_results = {}

        for method in all_results.keys():
            results = all_results[method]

            # Extract overall metrics
            kl_prob_values = [r.get('average_kl_prob', 0) for r in results if r]
            kl_argmax_values = [r.get('average_kl_argmax', 0) for r in results if r]
            accuracy_values = [r.get('average_accuracy', 0) for r in results if r and 'average_accuracy' in r]

            # Aggregate fraction-wise results
            fraction_wise_results = aggregate_fractionwise_kl(results)

            aggregated_results[method] = {
                'kl_transformed_mean_prob': np.mean(kl_prob_values) if kl_prob_values else 0.0,
                'kl_transformed_std_prob': np.std(kl_prob_values) if len(kl_prob_values) > 1 else 0.0,
                'kl_transformed_mean_onehot': np.mean(kl_argmax_values) if kl_argmax_values else 0.0,
                'kl_transformed_std_onehot': np.std(kl_argmax_values) if len(kl_argmax_values) > 1 else 0.0,
                'accuracy_transformed_mean': np.mean(accuracy_values) if accuracy_values else 0.0,
                'accuracy_transformed_std': np.std(accuracy_values) if len(accuracy_values) > 1 else 0.0,
                'fraction_wise_results_transformed': fraction_wise_results
            }

            # For baseline, also store as baseline results
            if method == 'baseline':
                aggregated_results[method].update({
                    'kl_baseline_mean_prob': np.mean(kl_prob_values) if kl_prob_values else 0.0,
                    'kl_baseline_std_prob': np.std(kl_prob_values) if len(kl_prob_values) > 1 else 0.0,
                    'kl_baseline_mean_onehot': np.mean(kl_argmax_values) if kl_argmax_values else 0.0,
                    'kl_baseline_std_onehot': np.std(kl_argmax_values) if len(kl_argmax_values) > 1 else 0.0,
                    'accuracy_baseline_mean': np.mean(accuracy_values) if accuracy_values else 0.0,
                    'accuracy_baseline_std': np.std(accuracy_values) if len(accuracy_values) > 1 else 0.0,
                    'fraction_wise_results': fraction_wise_results
                })

        return aggregated_results

    def build_kl_comparison_table(results):
        return str(results)

    def save_results(results, save_dir, dataset_name, n_runs=1):
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save with the correct filename for PhysioNet
        json_path = f"{save_dir}/physionet_results.json"
        with open(json_path, 'w') as f:
            import json
            # Convert numpy types to regular Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                     np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj

            serializable_results = convert_numpy(results)
            json.dump(serializable_results, f, indent=2)
        return json_path, None, None

    def convert_to_json_serializable(obj):
        return obj

    def plot_kl_divergence(results, save_path):
        pass

def aggregate_fractionwise_kl(fractionwise_results):
    """Aggregate fractionwise KL divergence and accuracy results across multiple runs."""
    if not fractionwise_results or not fractionwise_results[0]:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": [], "mean_accuracy": [], "std_accuracy": []}

    # Determine number of fractions from the first result
    first_result = fractionwise_results[0]
    if isinstance(first_result, dict) and 'kl_values_argmax' in first_result:
        num_fractions = len(first_result['kl_values_argmax'])
    else:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": [], "mean_accuracy": [], "std_accuracy": []}

    # Initialize arrays to store values for each fraction across runs
    kl_argmax_values = [[] for _ in range(num_fractions)]
    kl_prob_values = [[] for _ in range(num_fractions)]
    accuracy_values = [[] for _ in range(num_fractions)]

    # Collect values across all runs
    for run_results in fractionwise_results:
        kl_argmax_list = run_results['kl_values_argmax']
        kl_prob_list = run_results['kl_values_prob']

        for i in range(min(len(kl_argmax_list), num_fractions)):
            kl_argmax_values[i].append(kl_argmax_list[i])
            kl_prob_values[i].append(kl_prob_list[i])

        # Collect accuracy values if available
        if 'kl_values_accuracy' in run_results:
            accuracy_list = run_results['kl_values_accuracy']
            for i in range(min(len(accuracy_list), num_fractions)):
                accuracy_values[i].append(accuracy_list[i])

    # Calculate mean and standard deviation for each fraction
    mean_argmax = [np.mean(values) if values else 0.0 for values in kl_argmax_values]
    std_argmax = [np.std(values) if len(values) > 1 else 0.0 for values in kl_argmax_values]
    mean_prob = [np.mean(values) if values else 0.0 for values in kl_prob_values]
    std_prob = [np.std(values) if len(values) > 1 else 0.0 for values in kl_prob_values]
    mean_accuracy = [np.mean(values) if values else 0.0 for values in accuracy_values]
    std_accuracy = [np.std(values) if len(values) > 1 else 0.0 for values in accuracy_values]

    return {
        "mean_argmax": mean_argmax,
        "std_argmax": std_argmax,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy
    }


# MCal optimization utilities
from mcal.utils.optimization import kl_divergence, get_expectation

# Per-fraction and combined MCal_CE summaries are written here
MCAL_CE_RESULTS_DIR = Path(__file__).parent / "results"


def calculate_kl_metrics(outputs, labels=None, device=None):
    """
    Calculate KL divergence metrics using expectation approach.

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

    # Process labels once outside the loop
    processed_labels = None
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            processed_labels = torch.tensor(labels, dtype=torch.long, device=device)
        else:
            processed_labels = labels.to(device)

    # Calculate expectation for each fraction
    expectations_prob = []
    expectations_argmax = []
    accuracies = []
    kl_values_argmax = []
    kl_values_prob = []

    # Create uniform distribution for KL calculation
    uniform_dist = torch.full((n_outputs,), 1.0 / n_outputs, device=device)

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

        # Calculate fraction-wise KL divergences
        kl_argmax = kl_divergence(exp_argmax, uniform_dist).item()
        kl_prob = kl_divergence(exp_prob, uniform_dist).item()

        kl_values_argmax.append(kl_argmax)
        kl_values_prob.append(kl_prob)

        # Calculate accuracy if labels provided
        if processed_labels is not None:
            pred_labels = probs.argmax(dim=-1)
            accuracy = (pred_labels == processed_labels).float().mean().item()
            accuracies.append(accuracy)

    # Stack expectations
    expectations_prob = torch.stack(expectations_prob)  # (n_fractions, n_classes)
    expectations_argmax = torch.stack(expectations_argmax)  # (n_fractions, n_classes)

    # Calculate average KL divergences
    avg_kl_prob = np.mean(kl_values_prob)
    avg_kl_argmax = np.mean(kl_values_argmax)

    results = {
        'average_kl_prob': avg_kl_prob,
        'average_kl_argmax': avg_kl_argmax,
        'kl_values_argmax': kl_values_argmax,
        'kl_values_prob': kl_values_prob
    }

    if processed_labels is not None and accuracies:
        results['average_accuracy'] = np.mean(accuracies)
        results['kl_values_accuracy'] = accuracies  # Fractionwise accuracy values

    return results


def apply_transform(outputs, labels, method, device=None, **kwargs):
    """Apply a transformation method to outputs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Applying {method} transform...")

    if method == 'baseline' or method == 'retrain':
        return outputs

    elif method == 'mcal':
        return apply_mcal_calibrator(outputs, device, **kwargs)

    elif method == 'mcal_ce':
        return apply_mcal_ce_calibrator(outputs, labels, device, **kwargs)

    elif method == 'mcal_ce_uncond':
        return apply_mcal_ce_uncond_calibrator(outputs, labels, device, **kwargs)

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
    """Apply MCal calibrator."""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)

    # Create uniform target distribution
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
        save_fit_summary(calibrator, MCAL_CE_RESULTS_DIR)

        calibrated_probs = calibrator.forward(outputs_tensor[fraction])
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    # Combine results
    print(f"\n=== Combining MCal_CE results for experiment: {experiment_id} ===")
    combined_file = combine_fraction_results(MCAL_CE_RESULTS_DIR, experiment_id, cleanup_temp_files=True)
    if combined_file:
        print(f"All MCal_CE results combined and saved to: {combined_file}")

    return transformed_outputs


def apply_mcal_ce_uncond_calibrator(outputs_tensor, target_labels, device, max_steps=5000,
                                   head_type="linear", experiment_id="physionet_experiment", **kwargs):
    """Apply MCal_CE_Uncond calibrator using unconditional training approach."""
    # Convert numpy arrays to torch tensors if needed
    if not isinstance(outputs_tensor, torch.Tensor):
        outputs_tensor = torch.tensor(outputs_tensor, dtype=torch.float32, device=device)
    if not isinstance(target_labels, torch.Tensor):
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

    n_fractions, n_samples, n_classes = outputs_tensor.shape
    transformed_outputs = np.zeros_like(outputs_tensor.cpu().numpy())

    # Create training tensor by randomly sampling from all fractions
    train_tensor = torch.zeros_like(outputs_tensor[0])

    for i in range(n_samples):
        fraction_ind = np.random.binomial(n_fractions-1, 0.5)  # Random fraction selection
        train_tensor[i, :] = outputs_tensor[fraction_ind][i]

    # Use clean predictions (fraction 0) as target labels for training
    training_labels = outputs_tensor[0].argmax(dim=-1)

    # Create and fit single MCal_CE calibrator
    calibrator = MCal_CE(num_classes=n_classes, head_type=head_type)
    calibrator.to(device)
    calibrator.fit(
        ablated_probs=train_tensor,
        target_labels=training_labels,
        max_steps=max_steps,
        lr=1e-3,
        verbose=True,
        fraction=0,  # Use 0 as placeholder for unconditional training
        experiment_id=experiment_id
    )
    save_fit_summary(calibrator, MCAL_CE_RESULTS_DIR)

    # Apply the single calibrator to all fractions
    for fraction in tqdm(range(n_fractions), desc="Applying MCal_CE_Uncond calibrator"):
        calibrated_probs = calibrator.forward(outputs_tensor[fraction])
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    # Combine results
    print(f"\n=== Combining MCal_CE_Uncond results for experiment: {experiment_id} ===")
    combined_file = combine_fraction_results(MCAL_CE_RESULTS_DIR, experiment_id, cleanup_temp_files=True)
    if combined_file:
        print(f"All MCal_CE_Uncond results combined and saved to: {combined_file}")

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
    """Apply LogitsSharp transform."""
    print("⚠️  LogitsSharp transform not yet implemented for tabular data")
    return outputs


def apply_expectation_prob_transform(outputs, device):
    """Apply expectation probability transform."""
    print("⚠️  Expectation prob transform not implemented yet")
    return outputs


def apply_expectation_onehot_transform(outputs, device):
    """Apply expectation one-hot transform."""
    print("⚠️  Expectation one-hot transform not implemented yet")
    return outputs


def apply_optimized_lambda_transform(outputs, device, **kwargs):
    """Stub - apply optimized lambda transform."""
    print("⚠️  Optimized lambda transform not implemented yet")
    return outputs


def apply_archmod_transform(outputs, device, **kwargs):
    """Apply architecture modification transform using custom missing value (-10) handling."""
    return outputs  # No transformation - custom missing value handling applied during data generation


def process_physionet_dataset(methods=None, device="cuda", save_dir="./results", n_runs=3,
                            n_samples=1000, n_fractions=10):
    """Process PhysioNet dataset and generate KL benchmarks."""

    if methods is None:
        methods = ['baseline', 'mcal', 'mcal_ce', 'mcal_ce_uncond', 'platt', 'temperature', 'logits_sharp', 'retrain', 'replace', 'archmod']

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
        if any(method in ['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp','mcal_ce_uncond'] for method in methods):
            predictions, labels = load_physionet_data(
                model_type="vanilla",  # Clean, simple vanilla model
                fill_value="mean",     # Mean imputation for missing values
                n_samples=n_samples,
                n_fractions=n_fractions
            )
            print(f"Loaded vanilla data - Predictions: {predictions.shape}, Labels: {labels.shape}")

            # Store for all standard methods
            for method in ['baseline', 'mcal', 'mcal_ce','mcal_ce_uncond' ,'platt', 'temperature', 'logits_sharp']:
                if method in methods:
                    method_predictions[method] = predictions

        # Retrain method uses retrained model
        if 'retrain' in methods:
            retrain_predictions, retrain_labels = load_physionet_data(
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
            replace_predictions, replace_labels = load_physionet_data(
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
            archmod_predictions, archmod_labels = load_physionet_data(
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
                if method in ['mcal', 'mcal_ce','mcal_ce_uncond' 'platt', 'temperature']:
                    method_kwargs['max_steps'] = 1000
                if method == 'mcal':
                    method_kwargs['kappa'] = 10.0
                elif method in ['mcal_ce','mcal_ce_uncond']:
                    method_kwargs['max_steps'] = 5000
                    method_kwargs['head_type'] = 'linear'
                    method_kwargs['experiment_id'] = f'physionet_run_{run}'
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
    json_path, table_path, plot_path = save_results(aggregated_results, save_dir, "physionet", n_runs)

    # Also save to the expected location for plotting notebook
    import shutil
    expected_path = f"{save_dir}/physionet_results.json"
    if json_path != expected_path:
        shutil.copy2(json_path, expected_path)
        print(f"Results also saved to: {expected_path}")

    # Print summary table
    print("\nFinal Results Summary:")
    table = build_kl_comparison_table(aggregated_results)
    print(table)

    # Print fractionwise accuracy summary
    print("\nFractionwise Accuracy Summary:")
    for method, method_results in aggregated_results.items():
        if 'fraction_wise_results_transformed' in method_results:
            fwr = method_results['fraction_wise_results_transformed']
            if 'mean_accuracy' in fwr and any(acc > 0 for acc in fwr['mean_accuracy']):
                avg_accuracy = np.mean(fwr['mean_accuracy'])
                accuracy_std = np.mean(fwr['std_accuracy'])
                print(f"  {method.upper()}: Average fractionwise accuracy: {avg_accuracy:.4f} ± {accuracy_std:.4f}")
                if method_results.get('accuracy_transformed_mean', 0) > 0:
                    print(f"    Overall accuracy: {method_results['accuracy_transformed_mean']:.4f} ± {method_results['accuracy_transformed_std']:.4f}")

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
    parser = argparse.ArgumentParser(description="PhysioNet Tabular KL Divergence Benchmark")
    parser.add_argument("--methods", nargs="+",
                       choices=['baseline', 'mcal', 'mcal_ce', 'mcal_ce_uncond', 'platt', 'temperature', 'logits_sharp', 'retrain', 'replace', 'archmod'],
                       default=['baseline', 'mcal_ce', 'mcal_ce_uncond', 'retrain'],
                       help="Methods to benchmark")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--save_dir", default=str(Path(__file__).parent / "results"), help="Directory to save results")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples per run")
    parser.add_argument("--n_fractions", type=int, default=10, help="Number of ablation fractions")

    args = parser.parse_args()

    # Run benchmark
    results = process_physionet_dataset(
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