#!/usr/bin/env python3
"""
CTG (Cardiotocography) Multi-Class KL Divergence Benchmark for MCal

This script provides multi-class calibration evaluation for CTG fetal monitoring data,
extending MCal's framework to 3-class classification with medical-realistic missing data.

Classes: Normal (0), Suspect (1), Pathological (2)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import warnings

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import CTG data setup
from ctg_data_setup import load_ctg_data

# Import existing MCal components
try:
    from tabular_utils import aggregate_results, build_kl_comparison_table, save_results, plot_kl_divergence
except ImportError as e:
    print(f"Error: Could not import tabular_utils: {e}")
    raise e

# MCal calibrator imports
try:
    sys.path.append('/home/antonxue/shailesh/MCal/src')
    from calibrators.mcal import MCal
    from calibrators.mcal_ce import MCal_CE
    from calibrators.platt import PlattCalibrator
    from calibrators.temperature import TemperatureScaling
    from utils.optimization import kl_divergence, get_expectation
except ImportError as e:
    print(f"Warning: Could not import MCal calibrators: {e}")
    raise e

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


# ==================================================
# MULTI-CLASS CALIBRATION FUNCTIONS
# ==================================================

def softmax_temperature_calibration(outputs, temperature):
    """Apply temperature scaling to multi-class outputs."""
    return F.softmax(outputs / temperature, dim=-1)


def apply_multiclass_mcal_calibrator(outputs, device=None, **kwargs):
    """
    Apply MCal vector scaling calibrator to multi-class outputs.

    Args:
        outputs (np.ndarray): Model outputs of shape (n_samples, n_classes)
        device: PyTorch device
        **kwargs: Additional parameters

    Returns:
        np.ndarray: Calibrated outputs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to torch tensor
    if isinstance(outputs, np.ndarray):
        outputs_tensor = torch.from_numpy(outputs).float().to(device)
    else:
        outputs_tensor = outputs.float().to(device)

    # Use MCal calibrator - create uniform target distribution for multi-class
    try:
        n_samples, n_classes = outputs_tensor.shape
        target_dist = torch.ones(n_classes, device=device) / n_classes  # Uniform distribution
        target_dist = target_dist.unsqueeze(0).expand(n_samples, -1)

        calibrator = MCal(
            kappa=kwargs.get('kappa', 10.0),
            max_steps=kwargs.get('max_steps', 1000),
            device=device
        )

        # Apply MCal vector scaling
        calibrated = calibrator.calibrate(outputs_tensor, target_dist)

        return calibrated.cpu().numpy()

    except Exception as e:
        print(f"MCal calibration failed: {e}")
        raise e


def apply_multiclass_temperature_scaling(outputs, labels, device=None, **kwargs):
    """
    Apply temperature scaling to multi-class outputs.

    Args:
        outputs (np.ndarray): Model outputs of shape (n_fractions, n_samples, n_classes)
        labels (np.ndarray): True labels of shape (n_samples,)
        device: PyTorch device
        **kwargs: Additional parameters

    Returns:
        np.ndarray: Calibrated outputs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to torch tensors
    if isinstance(outputs, np.ndarray):
        outputs_tensor = torch.from_numpy(outputs).float().to(device)
    else:
        outputs_tensor = outputs.float().to(device)

    if labels is not None:
        if isinstance(labels, np.ndarray):
            labels_tensor = torch.from_numpy(labels).long().to(device)
        else:
            labels_tensor = labels.long().to(device)

    # Handle input dimensions
    if outputs_tensor.dim() == 3:
        n_fractions, n_samples, n_classes = outputs_tensor.shape
    else:
        # Handle 2D input for backward compatibility
        n_samples, n_classes = outputs_tensor.shape
        n_fractions = 1
        outputs_tensor = outputs_tensor.unsqueeze(0)

    transformed_outputs = torch.zeros_like(outputs_tensor)

    # Simple temperature scaling implementation
    try:
        from torch.optim import LBFGS

        class TemperatureScaler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = torch.nn.Parameter(torch.ones(1))

            def forward(self, logits):
                return F.softmax(logits / self.temperature, dim=-1)

        # Fit temperature scaler on fraction 0 (clean data)
        clean_outputs = outputs_tensor[0]  # Shape: (n_samples, n_classes)
        temp_scaler = TemperatureScaler().to(device)

        if labels is not None:
            # Optimize temperature on clean data and true labels
            optimizer = LBFGS(temp_scaler.parameters(), lr=0.01, max_iter=50)

            def eval_loss():
                optimizer.zero_grad()
                probs = temp_scaler(clean_outputs)
                loss = F.cross_entropy(torch.log(probs + 1e-8), labels_tensor)
                loss.backward()
                return loss

            optimizer.step(eval_loss)

        # Apply calibration to all fractions
        with torch.no_grad():
            for fraction in range(n_fractions):
                fraction_outputs = outputs_tensor[fraction]
                calibrated = temp_scaler(fraction_outputs)
                transformed_outputs[fraction] = calibrated

        # Return appropriate shape
        if n_fractions == 1:
            return transformed_outputs[0].cpu().numpy()
        else:
            return transformed_outputs.cpu().numpy()

    except Exception as e:
        print(f"Temperature scaling failed: {e}")
        raise e


def apply_multiclass_mcal_ce_calibrator(outputs, labels, device=None, **kwargs):
    """
    Apply MCal_CE calibrator to multi-class outputs using cross-entropy loss.

    Args:
        outputs (np.ndarray): Model outputs of shape (n_fractions, n_samples, n_classes)
        labels (np.ndarray): True labels of shape (n_samples,)
        device: PyTorch device
        **kwargs: Additional parameters

    Returns:
        np.ndarray: Calibrated outputs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Convert to torch tensors
        if isinstance(outputs, np.ndarray):
            outputs_tensor = torch.from_numpy(outputs).float().to(device)
        else:
            outputs_tensor = outputs.float().to(device)

        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels_tensor = torch.from_numpy(labels).long().to(device)
            else:
                labels_tensor = labels.long().to(device)

        # Get dimensions
        if outputs_tensor.dim() == 3:
            n_fractions, n_samples, n_classes = outputs_tensor.shape
        else:
            # Single fraction case
            n_samples, n_classes = outputs_tensor.shape
            n_fractions = 1
            outputs_tensor = outputs_tensor.unsqueeze(0)

        # Use clean distribution (0% missing) as target for all fractions
        target_labels = outputs_tensor[0].argmax(dim=-1)

        # Initialize result array
        transformed_outputs = np.zeros_like(outputs_tensor.cpu().numpy())

        print(f"Applying MCal_CE to {n_fractions} fractions with {n_classes} classes...")

        for fraction in tqdm(range(n_fractions), desc="Training MCal_CE calibrators"):
            try:
                # Create MCal_CE calibrator for this fraction
                calibrator = MCal_CE(
                    num_classes=n_classes,
                    head_type=kwargs.get('head_type', 'linear')
                )
                calibrator.to(device)

                # Train the calibrator
                calibrator.fit(
                    ablated_probs=outputs_tensor[fraction],
                    target_labels=target_labels,
                    max_steps=kwargs.get('max_steps', 1000),
                    lr=kwargs.get('lr', 0.001),
                    experiment_id=kwargs.get('experiment_id', f'ctg_mcal_ce_fraction_{fraction}')
                )

                # Apply calibration
                with torch.no_grad():
                    calibrated_probs = calibrator.forward(outputs_tensor[fraction])
                    transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

            except Exception as e:
                print(f"MCal_CE calibration failed for fraction {fraction}: {e}")
                raise e

        # Clean up if experiment_id was provided
        experiment_id = kwargs.get('experiment_id', None)
        if experiment_id:
            try:
                combined_file = MCal_CE.combine_fraction_results(experiment_id, cleanup_temp_files=True)
                if combined_file:
                    print(f"MCal_CE results combined and saved to: {combined_file}")
            except Exception as e:
                print(f"MCal_CE cleanup failed: {e}")
                raise e

        # Return appropriate shape
        if n_fractions == 1:
            return transformed_outputs[0]
        else:
            return transformed_outputs

    except Exception as e:
        raise e


def apply_multiclass_platt_scaling(outputs, labels, device=None, **kwargs):
    """
    Apply Platt scaling to multi-class outputs using one-vs-rest approach.

    Args:
        outputs (np.ndarray): Model outputs of shape (n_fractions, n_samples, n_classes)
        labels (np.ndarray): True labels of shape (n_samples,)
        device: PyTorch device
        **kwargs: Additional parameters

    Returns:
        np.ndarray: Calibrated outputs
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        # Handle 3D input
        if outputs.ndim == 3:
            n_fractions, n_samples, n_classes = outputs.shape
        else:
            # Handle 2D input for backward compatibility
            n_samples, n_classes = outputs.shape
            n_fractions = 1
            outputs = outputs.reshape(1, n_samples, n_classes)

        transformed_outputs = np.zeros_like(outputs)

        # Fit Platt scaling on fraction 0 (clean data)
        clean_outputs = outputs[0]  # Shape: (n_samples, n_classes)

        # Train one logistic regression per class
        platt_scalers = []
        for class_idx in range(n_classes):
            if labels is not None:
                binary_labels = (labels == class_idx).astype(int)
                lr = LogisticRegression()
                class_scores = clean_outputs[:, class_idx].reshape(-1, 1)

                try:
                    lr.fit(class_scores, binary_labels)
                    platt_scalers.append(lr)
                except Exception as e:
                    print(f"Platt scaling fit failed for class {class_idx}: {e}")
                    raise e
            else:
                platt_scalers.append(None)

        # Apply fitted Platt scalers to all fractions
        for fraction in range(n_fractions):
            fraction_outputs = outputs[fraction]
            calibrated_probs = np.zeros_like(fraction_outputs)

            for class_idx in range(n_classes):
                if platt_scalers[class_idx] is not None:
                    class_scores = fraction_outputs[:, class_idx].reshape(-1, 1)
                    calibrated_probs[:, class_idx] = platt_scalers[class_idx].predict_proba(class_scores)[:, 1]
                else:
                    calibrated_probs[:, class_idx] = fraction_outputs[:, class_idx]

            # Normalize to ensure valid probability distribution
            calibrated_probs = calibrated_probs / (calibrated_probs.sum(axis=1, keepdims=True) + 1e-8)
            transformed_outputs[fraction] = calibrated_probs

        # Return appropriate shape
        if n_fractions == 1:
            return transformed_outputs[0]
        else:
            return transformed_outputs

    except Exception as e:
        print(f"Platt scaling failed: {e}")
        raise e


def calculate_multiclass_kl_metrics(outputs, labels, device):
    """
    Calculate multi-class KL divergence metrics using mean expectation approach.

    Args:
        outputs (np.ndarray): Predictions of shape (n_fractions, n_samples, n_classes)
        labels (np.ndarray): True labels of shape (n_samples,)
        device: PyTorch device

    Returns:
        dict: KL divergence metrics
    """
    device = torch.device(device) if isinstance(device, str) else device
    n_fractions, n_samples, n_classes = outputs.shape

    # Convert to torch tensors
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32, device=device)

    # Get clean distribution (0% missing) expectations for reference
    clean_dist = outputs_tensor[0]
    clean_one_hot_expectation, clean_prob_expectation = get_expectation(clean_dist, device)

    # Calculate KL divergence for each fraction
    kl_probs = []
    kl_argmax = []
    accuracies = []

    for fraction in range(n_fractions):
        fraction_outputs = outputs_tensor[fraction]

        # Get expectations for this fraction
        one_hot_expectation, prob_expectation = get_expectation(fraction_outputs, device)

        # Calculate KL divergences between expectations and clean distribution
        kl_argmax_val = kl_divergence(one_hot_expectation, clean_one_hot_expectation).item()
        kl_prob = kl_divergence(prob_expectation, clean_prob_expectation).item()

        kl_probs.append(kl_prob)
        kl_argmax.append(kl_argmax_val)

        # Calculate accuracy for this fraction (fraction vs clean predictions)
        clean_predictions = torch.argmax(clean_dist, dim=1)
        fraction_predictions = torch.argmax(fraction_outputs, dim=1)
        accuracy = (clean_predictions == fraction_predictions).float().mean().item()
        accuracies.append(accuracy)

        print(f"Fraction {fraction}/{n_fractions} - KL Argmax: {kl_argmax_val:.6f}, KL Prob: {kl_prob:.6f}, Accuracy: {accuracy:.4f}")

    return {
        'kl_probs': kl_probs,
        'kl_argmax': kl_argmax,
        'accuracies': accuracies,
        'average_kl_prob': np.mean(kl_probs),
        'average_kl_argmax': np.mean(kl_argmax),
        'average_accuracy': np.mean(accuracies)
    }


def apply_transform(outputs, labels, method, device=None, **kwargs):
    """Apply calibration transform to multi-class outputs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Applying {method} transform...")

    if method == 'baseline':
        return outputs

    elif method == 'mcal':
        return apply_multiclass_mcal_calibrator(outputs, device, **kwargs)

    elif method == 'mcal_ce':
        return apply_multiclass_mcal_ce_calibrator(outputs, labels, device, **kwargs)

    elif method == 'temperature':
        return apply_multiclass_temperature_scaling(outputs, labels, device, **kwargs)

    elif method == 'platt':
        return apply_multiclass_platt_scaling(outputs, labels, device, **kwargs)

    else:
        raise ValueError(f"Unknown calibration method: {method}")


# ==================================================
# MAIN BENCHMARK FUNCTION
# ==================================================

def process_ctg_dataset(methods=None, device="cuda", save_dir="./results", n_runs=3,
                       n_samples=1000, n_fractions=10, ablation_strategy="medical"):
    """
    Process CTG dataset and generate multi-class KL benchmarks.

    Args:
        methods (list): Calibration methods to evaluate
        device (str): Computing device
        save_dir (str): Results save directory
        n_runs (int): Number of runs
        n_samples (int): Samples per run
        n_fractions (int): Number of missing data fractions
        ablation_strategy (str): Missing data ablation strategy

    Returns:
        dict: Aggregated results
    """
    if methods is None:
        methods = ['baseline', 'mcal', 'mcal_ce', 'temperature', 'platt']

    device = torch.device(device)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)

    print("="*60)
    print("CTG Multi-Class KL Divergence Benchmark")
    print("="*60)
    print(f"Methods: {methods}")
    print(f"Runs: {n_runs}")
    print(f"Samples per run: {n_samples}")
    print(f"Fractions: {n_fractions}")
    print(f"Ablation strategy: {ablation_strategy}")
    print(f"Device: {device}")

    # Initialize results storage
    all_results = {method: [] for method in methods}

    # Run multiple experiments
    for run in range(n_runs):
        print(f"\\n--- Run {run + 1}/{n_runs} ---")

        # Load CTG data
        predictions, labels = load_ctg_data(
            model_type="vanilla",
            n_samples=n_samples,
            n_fractions=n_fractions,
            ablation_strategy=ablation_strategy,
            balanced=True
        )

        print(f"Loaded CTG data - Predictions: {predictions.shape}, Labels: {labels.shape}")

        # Process each method
        for method in methods:
            print(f"\\nProcessing method: {method}")

            # Apply transformation
            if method == 'baseline':
                transformed_predictions = predictions
            else:
                # Configure method-specific parameters
                method_kwargs = {}
                if method in ['mcal', 'mcal_ce', 'temperature', 'platt']:
                    method_kwargs['max_steps'] = 1000
                if method == 'mcal':
                    method_kwargs['kappa'] = 10.0
                elif method == 'mcal_ce':
                    method_kwargs['max_steps'] = 5000
                    method_kwargs['head_type'] = 'linear'
                    method_kwargs['experiment_id'] = f'ctg_run_{run}'

                transformed_predictions = apply_transform(
                    predictions.numpy(), labels.numpy(), method, device, **method_kwargs
                )
                transformed_predictions = torch.from_numpy(transformed_predictions)

            # Calculate multi-class KL metrics
            kl_results = calculate_multiclass_kl_metrics(
                transformed_predictions.numpy(), labels.numpy(), device
            )
            all_results[method].append(kl_results)

            print(f"  KL (prob): {kl_results['average_kl_prob']:.6f}")
            print(f"  KL (argmax): {kl_results['average_kl_argmax']:.6f}")
            print(f"  Average accuracy: {kl_results['average_accuracy']:.4f}")

    print("\\nAggregating results across all runs...")

    # Aggregate results across runs
    try:
        aggregated_results = aggregate_ctg_results(all_results)

        # Save results
        results_file = os.path.join(save_dir, "json", "aggregated_results_ctg.json")
        save_ctg_results(aggregated_results, results_file)

        # Generate comparison table
        table = build_ctg_comparison_table(aggregated_results)
        print(f"\\n{table}")

        # Save table
        table_file = os.path.join(save_dir, "kl_comparison_table_ctg.txt")
        with open(table_file, 'w') as f:
            f.write(table)
        print(f"Comparison table saved to {table_file}")

        # Plot results if possible
        try:
            plot_file = os.path.join(save_dir, "kl_divergence_ctg.png")
            plot_ctg_kl_divergence(aggregated_results, plot_file)
            print(f"Plot saved to {plot_file}")
        except Exception as e:
            print(f"Plotting failed: {e}")
            raise e

    except Exception as e:
        print(f"Results processing failed: {e}")
        raise e

    print("\\nCTG benchmark completed! 🎉")

    # Print summary
    print("\\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for method in methods:
        if method in aggregated_results and aggregated_results[method]:
            if isinstance(aggregated_results[method], list):
                # Use first run if aggregation failed
                result = aggregated_results[method][0]
                avg_kl_prob = result['average_kl_prob']
                avg_kl_argmax = result['average_kl_argmax']
            else:
                # Use aggregated results
                avg_kl_prob = aggregated_results[method].get('average_kl_prob', 0)
                avg_kl_argmax = aggregated_results[method].get('average_kl_argmax', 0)

            print(f"{method.upper()}:")
            print(f"  KL Prob: {avg_kl_prob:.2e}")
            print(f"  KL Argmax: {avg_kl_argmax:.2e}")

    return aggregated_results


# ==================================================
# RESULTS UTILITIES (CTG-specific)
# ==================================================

def aggregate_ctg_results(all_results):
    """Aggregate CTG results across multiple runs."""
    aggregated = {}

    for method, runs in all_results.items():
        if not runs:
            continue

        # Aggregate metrics across runs
        kl_probs = [run['average_kl_prob'] for run in runs]
        kl_argmax = [run['average_kl_argmax'] for run in runs]
        accuracies = [run['average_accuracy'] for run in runs]

        aggregated[method] = {
            'kl_prob_mean': np.mean(kl_probs),
            'kl_prob_std': np.std(kl_probs),
            'kl_argmax_mean': np.mean(kl_argmax),
            'kl_argmax_std': np.std(kl_argmax),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies)
        }

    return aggregated


def build_ctg_comparison_table(aggregated_results):
    """Build comparison table for CTG results."""
    table_lines = []
    table_lines.append("CTG Multi-Class KL Divergence Comparison:")
    table_lines.append("+----------------------+---------------------+---------------------+")
    table_lines.append("| Method               | Average KL (Prob)  | Average KL (Argmax)|")
    table_lines.append("+======================+=====================+=====================+")

    method_names = {
        'baseline': 'Baseline',
        'mcal': 'MCal Multi-Class',
        'mcal_ce': 'MCal_CE Multi-Class',
        'temperature': 'Temperature Scaling',
        'platt': 'Platt Multi-Class'
    }

    for method, data in aggregated_results.items():
        name = method_names.get(method, method.title())
        kl_prob = f"{data['kl_prob_mean']:.2e} ± {data['kl_prob_std']:.2e}"
        kl_argmax = f"{data['kl_argmax_mean']:.2e} ± {data['kl_argmax_std']:.2e}"

        table_lines.append(f"| {name:<20} | {kl_prob:<19} | {kl_argmax:<19} |")

    table_lines.append("+----------------------+---------------------+---------------------+")

    return "\n".join(table_lines)


def save_ctg_results(results, file_path):
    """Save CTG results to JSON file."""
    import json
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Aggregated results saved to {file_path}")


def plot_ctg_kl_divergence(results, save_path):
    """Plot CTG KL divergence results."""
    try:
        import matplotlib.pyplot as plt

        methods = list(results.keys())
        kl_probs = [results[m]['kl_prob_mean'] for m in methods]
        kl_argmax = [results[m]['kl_argmax_mean'] for m in methods]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # KL Prob plot
        ax1.bar(methods, kl_probs)
        ax1.set_title('KL Divergence (Probability)')
        ax1.set_ylabel('KL Divergence')
        ax1.tick_params(axis='x', rotation=45)

        # KL Argmax plot
        ax2.bar(methods, kl_argmax)
        ax2.set_title('KL Divergence (Argmax)')
        ax2.set_ylabel('KL Divergence')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    except ImportError as e:
        print(f"Matplotlib import failed: {e}")
        raise e
    except Exception as e:
        print(f"Plotting failed: {e}")
        raise e


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CTG Multi-Class KL Divergence Benchmark")
    parser.add_argument("--methods", nargs='+',
                       default=['baseline', 'mcal', 'mcal_ce', 'temperature', 'platt'],
                       help="Methods to include in benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per run")
    parser.add_argument("--fractions", type=int, default=10, help="Number of fractions")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="./results", help="Save directory")
    parser.add_argument("--ablation_strategy", type=str, default="medical",
                       choices=["random", "equipment", "temporal", "protocol", "medical"],
                       help="Missing data ablation strategy")

    args = parser.parse_args()

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Run benchmark
    aggregated_results = process_ctg_dataset(
        methods=args.methods,
        device=device,
        save_dir=args.save_dir,
        n_runs=args.runs,
        n_samples=args.samples,
        n_fractions=args.fractions,
        ablation_strategy=args.ablation_strategy
    )

    return aggregated_results


if __name__ == "__main__":
    main()