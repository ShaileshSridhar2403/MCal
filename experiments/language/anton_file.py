#!/usr/bin/env python3
"""
MedMCQA KL Divergence Benchmark - MCal Implementation

Following the exact pattern of MedQA KL benchmarks (medqa_kl_benchmark.py)
but for MedMCQA dataset and LLaMA predictions.

Self-contained implementation - no XAI_Benchmark dependencies.
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
import pdb

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))
sys.path.insert(0, str(mcal_root / "configs"))

# Import MCal modules for transforms
from src.calibrators.mcal import MCal
from src.calibrators.mcal_ce import MCal_CE
from src.calibrators.temperature import TemperatureScaling
from src.calibrators.platt import PlattCalibrator

# Import our self-contained MedMCQA utilities
from medmcqa_utils import (
    MCal_LLaMAModel,
    load_local_medmcqa_data,
    load_synthetic_medmcqa_data,
    generate_fractionwise_predictions,
    generate_fractionwise_predictions_with_token_dropping,
    generate_fractionwise_predictions_with_attention_mask,
    create_medmcqa_prompt,
    map_probs_to_list
)

def load_medmcqa_llama_model(model_path):
    """Load LLaMA model for MedMCQA."""
    print(f"Loading LLaMA model from: {model_path}")

    model = MCal_LLaMAModel(model_path)
    return model

def calculate_kl_metrics(outputs, labels=None, device=None):
    """Calculate KL divergence metrics and accuracy for outputs."""
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

        # Uniform distribution for comparison
        # uniform_dist = torch.ones(n_outputs, device=device) / n_outputs
        # pdb.set_trace()
        # clean_dist = labels.mean(dim = 0)
        clean_dist = torch.tensor(outputs[0], dtype=torch.float32, device=device).mean(dim=0)

        
        # Calculate KL divergences
        # kl_argmax = kl_divergence(one_hot_expectation, uniform_dist).item()
        # kl_prob = kl_divergence(prob_expectation, uniform_dist).item()
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

            # Ensure both arrays are numpy arrays for comparison
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

    if method == 'baseline':
        return outputs
    elif method == 'mcal':
        return apply_mcal_calibrator(outputs, device, **kwargs)
    elif method == 'mcal_ce':
        return apply_mcal_ce_calibrator(outputs, labels, device, **kwargs)
    elif method == 'platt':
        return apply_platt_scaling(outputs, labels, device, **kwargs)
    elif method == 'temperature':
        return apply_temperature_scaling(outputs, labels, device, **kwargs)
    else:
        raise ValueError(f"Unknown transformation method: {method}")

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

        # Transform this fraction's predictions
        transformed_probs = calibrator.forward(ablated_probs)
        transformed_outputs[fraction] = transformed_probs.detach().cpu().numpy()

    return transformed_outputs

def apply_mcal_ce_calibrator(outputs_tensor, target_labels, device, max_steps=10000, head_type="linear", experiment_id="medmcqa_experiment", **kwargs):
    """Apply MCal_CE calibrator using cross-entropy loss."""
    # Convert numpy arrays to torch tensors if needed
    if not isinstance(outputs_tensor, torch.Tensor):
        outputs_tensor = torch.tensor(outputs_tensor, dtype=torch.float32, device=device)
    if not isinstance(target_labels, torch.Tensor):
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

    n_fractions, n_samples, n_classes = outputs_tensor.shape
    transformed_outputs = np.zeros_like(outputs_tensor.cpu().numpy())

    for fraction in tqdm(range(n_fractions), desc="Applying MCal_CE calibrator"):
        calibrator = MCal_CE(num_classes=n_classes, head_type="mlp")
        calibrator.to(device)
        # pdb.set_trace()
        calibrator.fit(
            ablated_probs=outputs_tensor[fraction],
            target_labels=target_labels,
            max_steps=max_steps,
            lr=1e-3,
            verbose=True,
            fraction=fraction,
            experiment_id=experiment_id
        )

        calibrated_probs = calibrator.forward(outputs_tensor[fraction])
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    # Combine results
    print(f"\n=== Combining MCal_CE results for experiment: {experiment_id} ===")
    combined_file = MCal_CE.combine_fraction_results(experiment_id, cleanup_temp_files=True)
    print(f"All MCal_CE results combined and saved to: {combined_file}")

    return transformed_outputs

def apply_platt_scaling(outputs_tensor, target_labels, device, **kwargs):
    """Apply Platt scaling."""
    n_fractions, n_samples, n_classes = outputs_tensor.shape
    transformed_outputs = np.zeros_like(outputs_tensor)

    for fraction in tqdm(range(n_fractions), desc="Applying Platt scaling"):
        calibrator = PlattCalibrator(num_classes=4)

        # Convert to tensors
        predictions = torch.from_numpy(outputs_tensor[fraction]).float()
        labels = torch.from_numpy(target_labels).long()

        # Fit and transform
        transformed_predictions = calibrator.fit_transform(predictions, labels)
        transformed_outputs[fraction] = transformed_predictions.numpy()

    return transformed_outputs

def apply_temperature_scaling(outputs_tensor, target_labels, device, **kwargs):
    """Apply temperature scaling."""
    n_fractions, n_samples, n_classes = outputs_tensor.shape
    transformed_outputs = np.zeros_like(outputs_tensor)

    for fraction in tqdm(range(n_fractions), desc="Applying temperature scaling"):
        calibrator = TemperatureScaling()

        # Convert to tensors
        predictions = torch.from_numpy(outputs_tensor[fraction]).float()
        labels = torch.from_numpy(target_labels).long()

        # Fit and transform
        transformed_predictions = calibrator.fit_transform(predictions, labels)
        transformed_outputs[fraction] = transformed_predictions.numpy()

    return transformed_outputs

def get_expectation(predictions, device):
    """Get expectation vectors for KL divergence calculation."""
    predictions = predictions.to(device)

    # One-hot expectation (argmax)
    one_hot = torch.zeros_like(predictions)
    argmax_indices = torch.argmax(predictions, dim=1)
    one_hot.scatter_(1, argmax_indices.unsqueeze(1), 1.0)
    one_hot_expectation = torch.mean(one_hot, dim=0)

    # Probability expectation (soft)
    prob_expectation = torch.mean(predictions, dim=0)

    return one_hot_expectation, prob_expectation

def kl_divergence(p, q, epsilon=1e-8):
    """Calculate KL divergence between two distributions."""
    p = torch.clamp(p, epsilon, 1.0)
    q = torch.clamp(q, epsilon, 1.0)
    return torch.sum(p * torch.log(p / q))

def load_medmcqa_data(model_type="vanilla", n_samples=10, n_fractions=10,
                     model_path="~/foo/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                     use_real_data=True, balanced=True):
    """Load MedMCQA data following vision benchmark pattern."""

    print(f"Loading MedMCQA data with {n_samples} samples, {n_fractions} fractions...")
    print(f"Real data: {use_real_data}, Balanced: {balanced}")

    if model_type in ["vanilla", "token_drop", "attention_mask", "qlora"]:
        print("THE MODEL TYPE IS", model_type)
        # Load model
        expanded_path = Path(model_path).expanduser()
        model = load_medmcqa_llama_model(str(expanded_path))

        # Load MedMCQA data (local or synthetic)
        if use_real_data:
            medmcqa_questions = load_local_medmcqa_data(n_samples, balanced=balanced)
        else:
            print("Using synthetic MedMCQA data...")
            medmcqa_questions = load_synthetic_medmcqa_data(n_samples)

        # Generate predictions with different ablation fractions
        removal_fractions = np.linspace(0, 0.9, n_fractions).tolist()

        if model_type == "token_drop":
            # Use token dropping strategy
            predictions = generate_fractionwise_predictions_with_token_dropping(
                model=model,
                data=medmcqa_questions,
                removal_fractions=removal_fractions,
                prompt_type='default',
                batch_size=min(8, n_samples),
                num_options=4,
                use_tokenizer=True
            )
        elif model_type == "attention_mask":
            # Use attention masking strategy (content-only)
            predictions = generate_fractionwise_predictions_with_attention_mask(
                model=model,
                data=medmcqa_questions,
                removal_fractions=removal_fractions,
                prompt_type='default',
                batch_size=min(8, n_samples),
                num_options=4
            )
        elif model_type == "qlora":
            # Use standard word replacement strategy

            expanded_path = Path("~/foo/MCal/saved_models/medmcqa/medmcqa_p0.0/merged_model").expanduser()
            model = load_medmcqa_llama_model(str(expanded_path))
            print(f"Using qlora merged model at: {expanded_path}")

            predictions = generate_fractionwise_predictions(
                model=model,
                data=medmcqa_questions,
                removal_fractions=removal_fractions,
                prompt_type='default',
                batch_size=min(8, n_samples),
                num_options=4
            )
            
        else:
            predictions = generate_fractionwise_predictions(
                model=model,
                data=medmcqa_questions,
                removal_fractions=removal_fractions,
                prompt_type='default',
                batch_size=min(8, n_samples),
                num_options=4
            )

        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Generate labels from correct answers (common to both vanilla and token_drop)
    if use_real_data and 'cop' in medmcqa_questions[0]:
        # Use actual correct answers for real data
        labels = np.array([q['cop'] - 1 for q in medmcqa_questions])  # Convert 1-indexed to 0-indexed
        print(f"Using real MedMCQA ground truth labels")

        # Report label distribution
        label_dist = {i: np.sum(labels == i) for i in range(4)}
        label_dist_letters = {chr(65 + i): count for i, count in label_dist.items()}
        print(f"Label distribution: {label_dist_letters}")
    else:
        # Fallback to argmax of clean predictions for synthetic data
        clean_predictions = predictions[0]  # First fraction (no ablation)
        labels = np.argmax(clean_predictions, axis=1)
        print(f"Using argmax of clean predictions as labels")

    print(f"Generated predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")

    # Convert to torch tensors
    predictions = torch.from_numpy(predictions).float()
    labels = torch.from_numpy(labels).long()

    return predictions, labels

def aggregate_results(all_results):
    """Aggregate results across multiple runs."""
    aggregated = {}

    for method, results_list in all_results.items():
        # Calculate mean and std for each metric
        kl_prob_values = [r['average_kl_prob'] for r in results_list]
        kl_argmax_values = [r['average_kl_argmax'] for r in results_list]

        # Aggregate fraction-wise results
        fraction_wise_results = {
            'mean_prob': [],
            'std_prob': [],
            'mean_argmax': [],
            'std_argmax': []
        }

        n_fractions = len(results_list[0]['kl_values_prob'])
        for fraction in range(n_fractions):
            prob_values = [r['kl_values_prob'][fraction] for r in results_list]
            argmax_values = [r['kl_values_argmax'][fraction] for r in results_list]

            fraction_wise_results['mean_prob'].append(np.mean(prob_values))
            fraction_wise_results['std_prob'].append(np.std(prob_values))
            fraction_wise_results['mean_argmax'].append(np.mean(argmax_values))
            fraction_wise_results['std_argmax'].append(np.std(argmax_values))

        if method == 'baseline':
            aggregated[method] = {
                'kl_baseline_mean_prob': np.mean(kl_prob_values),
                'kl_baseline_std_prob': np.std(kl_prob_values),
                'kl_baseline_mean_onehot': np.mean(kl_argmax_values),
                'kl_baseline_std_onehot': np.std(kl_argmax_values),
                'fraction_wise_results': fraction_wise_results
            }
        else:
            aggregated[method] = {
                'kl_transformed_mean_prob': np.mean(kl_prob_values),
                'kl_transformed_std_prob': np.std(kl_prob_values),
                'kl_transformed_mean_onehot': np.mean(kl_argmax_values),
                'kl_transformed_std_onehot': np.std(kl_argmax_values),
                'fraction_wise_results_transformed': fraction_wise_results
            }

    return aggregated

def save_results(results, save_dir="./results"):
    """Save results to JSON and generate comparison table."""
    # Create directories
    os.makedirs(f"{save_dir}/json", exist_ok=True)

    # Save to JSON
    json_file = f"{save_dir}/json/aggregated_results_medmcqa.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Aggregated results saved to {json_file}")

    # Generate comparison table
    table_data = []
    headers = ["Method", "Average KL (Prob)", "Average KL (Argmax)"]

    for method, data in results.items():
        if method == 'baseline':
            method_name = "Original"
            kl_prob = data['kl_baseline_mean_prob']
            kl_prob_std = data['kl_baseline_std_prob']
            kl_argmax = data['kl_baseline_mean_onehot']
            kl_argmax_std = data['kl_baseline_std_onehot']
        else:
            method_name = {
                'mcal': 'MCal',
                'mcal_ce': 'MCal_CE (Cross-Entropy)',
                'platt': 'Platt Scaling',
                'temperature': 'Temperature Scaling',
                'token_drop': 'Token Dropping',
                'attention_mask': 'Attention Masking',
                "qlora": "QLoRA"
            }.get(method, method.upper())
            kl_prob = data['kl_transformed_mean_prob']
            kl_prob_std = data['kl_transformed_std_prob']
            kl_argmax = data['kl_transformed_mean_onehot']
            kl_argmax_std = data['kl_transformed_std_onehot']

        table_data.append([
            method_name,
            f"{kl_prob:.2e} ± {kl_prob_std:.2e}",
            f"{kl_argmax:.2e} ± {kl_argmax_std:.2e}"
        ])

    # Generate table
    table = tabulate(table_data, headers=headers, tablefmt="grid")

    # Save table
    table_file = f"{save_dir}/kl_comparison_table_medmcqa.txt"

    # Get number of runs from fraction-wise results (which contain arrays of length = n_runs)
    first_method = list(results.keys())[0]
    if 'baseline' in results:
        n_runs = len(results['baseline']['fraction_wise_results']['mean_prob'])
    else:
        # Use the first available method's fraction-wise results
        first_method_data = results[first_method]
        if 'fraction_wise_results_transformed' in first_method_data:
            n_runs = len(first_method_data['fraction_wise_results_transformed']['mean_prob'])
        else:
            n_runs = len(first_method_data['fraction_wise_results']['mean_prob'])

    with open(table_file, 'w') as f:
        f.write(f"KL Divergence Comparison for MedMCQA (averaged over {n_runs} runs):\n")
        f.write(table)

    print(f"Comparison table saved to {table_file}")
    print(f"\nKL Divergence Comparison for MedMCQA:")
    print(table)

    return json_file, table_file

def process_medmcqa_dataset(methods=None, device="cuda", save_dir="./results", n_runs=3,
                           n_samples=10, n_fractions=10,
                           model_path="~/foo/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                           use_real_data=True, balanced=True):
    """Process MedMCQA dataset with multiple calibration methods."""

    if methods is None:
        methods = ['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature']

    # Storage for all results
    all_results = {method: [] for method in methods}

    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")

        # Check which data types we need
        need_token_drop = 'token_drop' in methods
        need_attention_mask = 'attention_mask' in methods
        need_vanilla = any(method not in ['token_drop', 'attention_mask', 'qlora'] for method in methods)

        # Load data for different ablation strategies
        all_predictions = {}
        all_labels = {}

        if need_vanilla:
            # Load vanilla (word replacement) data
            predictions_vanilla, labels_vanilla = load_medmcqa_data(
                model_type="vanilla",
                n_samples=n_samples,
                n_fractions=n_fractions,
                model_path=model_path,
                use_real_data=use_real_data,
                balanced=balanced
            )
            all_predictions['vanilla'] = predictions_vanilla
            all_labels['vanilla'] = labels_vanilla

        if need_token_drop:
            # Load token dropping data
            predictions_token_drop, labels_token_drop = load_medmcqa_data(
                model_type="token_drop",
                n_samples=n_samples,
                n_fractions=n_fractions,
                model_path=model_path,
                use_real_data=use_real_data,
                balanced=balanced
            )
            all_predictions['token_drop'] = predictions_token_drop
            all_labels['token_drop'] = labels_token_drop

        if need_attention_mask:
            # Load attention masking data
            predictions_attention_mask, labels_attention_mask = load_medmcqa_data(
                model_type="attention_mask",
                n_samples=n_samples,
                n_fractions=n_fractions,
                model_path=model_path,
                use_real_data=use_real_data,
                balanced=balanced
            )
            all_predictions['attention_mask'] = predictions_attention_mask
            all_labels['attention_mask'] = labels_attention_mask

        # Process each method
        for method in methods:
            print(f"\nProcessing method: {method}")

            # Select appropriate predictions and labels
            if method == 'token_drop':
                predictions = all_predictions['token_drop']
                labels = all_labels['token_drop']
            elif method == 'attention_mask':
                predictions = all_predictions['attention_mask']
                labels = all_labels['attention_mask']
            elif method == 'qlora':
                predictions = all_predictions['qlora']
                labels = all_labels['qlora']
            else:
                predictions = all_predictions['vanilla']
                labels = all_labels['vanilla']

            # Apply transformation
            if method in ['baseline', 'token_drop', 'attention_mask', 'qlora']:
                # For baseline, token_drop, and attention_mask: no additional transformation needed
                transformed_predictions = predictions.numpy()
            else:
                print(f"Applying {method} transform...")
                # Get method-specific kwargs
                method_kwargs = {}
                if method == 'mcal_ce':
                    method_kwargs['max_steps'] = 10000
                    method_kwargs['experiment_id'] = f"medmcqa_experiment"
                    # method_kwargs['lr'] = 1e-2
                
                labels = predictions[0].argmax(dim=-1)
                transformed_predictions = apply_transform(
                    predictions.numpy(), labels.numpy(), method, device, **method_kwargs
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
    save_results(aggregated_results, save_dir)

    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description="MedMCQA KL Divergence Benchmark")
    parser.add_argument("--methods", nargs='+', default=['baseline', 'mcal_ce'],
                       choices=['baseline', 'mcal', 'mcal_ce', 'platt', 'temperature', 'token_drop', 'attention_mask', 'qlora'],
                       help="Calibration methods to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs")
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of samples per run")
    parser.add_argument("--fractions", type=int, default=10,
                       help="Number of ablation fractions")
    parser.add_argument("--model_path", type=str,
                       default="~/foo/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                       help="Path to LLaMA model")
    parser.add_argument("--use_real_data", action="store_true", default=True,
                       help="Use real MedMCQA dataset (default: True)")
    parser.add_argument("--use_synthetic_data", action="store_true", default=False,
                       help="Use synthetic MedMCQA data instead of real data")
    parser.add_argument("--balanced", action="store_true", default=True,
                       help="Ensure balanced answer distribution (default: True)")

    args = parser.parse_args()

    # Handle data source flags
    use_real_data = args.use_real_data and not args.use_synthetic_data

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("=" * 60)
    print("MedMCQA KL Divergence Benchmark")
    print("=" * 60)
    print(f"Methods: {args.methods}")
    print(f"Runs: {args.runs}")
    print(f"Samples per run: {args.samples}")
    print(f"Fractions: {args.fractions}")
    print(f"Device: {device}")
    print()

    # Run benchmark
    aggregated_results = process_medmcqa_dataset(
        methods=args.methods,
        device=device,
        save_dir=args.save_dir,
        n_runs=args.runs,
        n_samples=args.samples,
        n_fractions=args.fractions,
        model_path=args.model_path,
        use_real_data=use_real_data,
        balanced=args.balanced
    )

    print("\nBenchmark completed! 🎉")

if __name__ == "__main__":
    main()