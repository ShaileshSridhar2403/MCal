#!/usr/bin/env python3
"""
MedQA QLoRA KL Divergence Benchmark - MCal Implementation

Integrates QLoRA models trained on binomially ablated data into the existing
MedQA KL divergence benchmark framework.
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

# Import MCal utilities
from src.utils.optimization import get_expectation, make_one_hot, kl_divergence

# Import calibrator modules
from src.calibrators.mcal import MCal
from src.calibrators.mcal_ce import MCal_CE
from src.calibrators.platt import PlattCalibrator
from src.calibrators.temperature import TemperatureScaling

# Import QLoRA utilities
from qlora_utils import MCal_QLoRA_Model, map_probs_to_list

# Import MedQA utilities for data loading
from medqa_utils import (
    load_local_medqa_data,
    create_medqa_prompt,
    generate_fractionwise_predictions,
    generate_fractionwise_predictions_with_attention_mask,
    generate_fractionwise_predictions_with_token_dropping
)

def load_qlora_model(base_model_path, lora_adapter_path):
    """Load QLoRA model for MedQA predictions."""
    print(f"Loading QLoRA model from: {base_model_path}")
    print(f"With LoRA adapters: {lora_adapter_path}")

    model = MCal_QLoRA_Model(base_model_path, lora_adapter_path)
    return model

def generate_fractionwise_predictions_qlora(
    model, data, removal_fractions, prompt_type='default', batch_size=8, num_options=5
):
    """
    Generate predictions using QLoRA model on clean test data.

    Note: QLoRA model was trained on ablated data, so we test its robustness
    on clean data and optionally on various ablation levels.
    """
    all_fraction_probs = []

    print(f"Using QLoRA model trained on binomially ablated data (0-90%)")
    print(f"Testing on clean and ablated data")

    for removal_fraction in tqdm(removal_fractions, desc="Processing removal fractions"):
        fraction_probs = []

        # Process data in batches
        for i in tqdm(range(0, len(data), batch_size),
                     desc=f"Processing questions (removal fraction: {removal_fraction:.1f})",
                     unit="batch", leave=False):
            batch = data[i:i+batch_size]

            # Process each item in the batch
            for question_data in batch:
                if removal_fraction == 0.0:
                    # Clean test data
                    prompt = create_medqa_prompt(
                        question_data,
                        removal_fraction=0.0,
                        prompt_type=prompt_type
                    )
                else:
                    # Apply ablation for robustness testing
                    prompt = create_medqa_prompt(
                        question_data,
                        removal_fraction=removal_fraction,
                        prompt_type=prompt_type
                    )

                # Debug: Show example prompts for small datasets
                if len(data) <= 5 and removal_fraction == 0:
                    print(f"\n=== QLORA BASELINE FRACTION=0 DEBUG ===")
                    print(f"Clean prompt (first item): {prompt[:200]}...")
                    print(f"Model type: QLoRA binomial ablated")
                    print("=== END QLORA DEBUG ===\n")
                elif len(data) <= 5 and removal_fraction > 0:
                    print(f"\n=== QLORA ABLATION FRACTION={removal_fraction} DEBUG ===")
                    print(f"Ablated prompt (first item): {prompt[:200]}...")
                    print(f"Test ablation fraction: {removal_fraction}")
                    print("=== END QLORA ABLATION DEBUG ===\n")

                # Get probabilities from QLoRA model
                probs = model.get_choice_probabilities(prompt, num_options)

                # Convert to list format
                prob_list = map_probs_to_list(probs, num_options=num_options)

                # Handle NaN values
                if np.isnan(np.array(prob_list)).any():
                    # Use uniform distribution as fallback
                    fraction_probs.append(np.ones(num_options) * (1/num_options))
                    continue

                fraction_probs.append(prob_list)

        all_fraction_probs.append(np.array(fraction_probs))

        # Print mean probabilities for this fraction
        if len(fraction_probs) > 0:
            mean_probs = np.mean(fraction_probs, axis=0)
            print(f"  Fraction {removal_fraction:.1f} - Mean probabilities: {mean_probs}")

    # Convert to numpy array with shape (n_fractions, n_samples, n_options)
    all_fraction_probs_np = np.array(all_fraction_probs)

    return all_fraction_probs_np

def load_medqa_data_qlora(lora_adapter_path, n_samples=10, n_fractions=10,
                         base_model_path="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                         use_real_data=True, balanced=True):
    """Load MedQA data with QLoRA model following existing benchmark pattern."""

    print(f"Loading MedQA data with QLoRA model: {n_samples} samples, {n_fractions} fractions...")
    print(f"Real data: {use_real_data}, Balanced: {balanced}")

    # Load QLoRA model
    expanded_base_path = Path(base_model_path).expanduser()
    expanded_adapter_path = Path(lora_adapter_path).expanduser()
    model = load_qlora_model(str(expanded_base_path), str(expanded_adapter_path))

    # Load MedQA data
    if use_real_data:
        medqa_questions = load_local_medqa_data(n_samples, balanced=balanced)
    else:
        raise ValueError("Synthetic data not supported for QLoRA benchmark")

    # Generate predictions with different test ablation fractions
    removal_fractions = np.linspace(0, 0.9, n_fractions).tolist()

    predictions = generate_fractionwise_predictions_qlora(
        model=model,
        data=medqa_questions,
        removal_fractions=removal_fractions,
        prompt_type='default',
        batch_size=min(8, n_samples),
        num_options=5
    )

    # Generate labels from correct answers
    if use_real_data and 'answer_idx' in medqa_questions[0]:
        # Use actual correct answers for real data
        labels = np.array([ord(q['answer_idx']) - ord('A') for q in medqa_questions])
        print(f"Using real MedQA ground truth labels")

        # Report label distribution
        label_dist = {i: np.sum(labels == i) for i in range(5)}
        label_dist_letters = {chr(65 + i): count for i, count in label_dist.items()}
        print(f"Label distribution: {label_dist_letters}")
    else:
        # Fallback to argmax of clean predictions
        clean_predictions = predictions[0]  # First fraction (no ablation)
        labels = np.argmax(clean_predictions, axis=1)
        print(f"Using argmax of clean predictions as labels")

    print(f"Generated predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")

    # Convert to torch tensors
    predictions = torch.from_numpy(predictions).float()
    labels = torch.from_numpy(labels).long()

    return predictions, labels

# ===== KL DIVERGENCE UTILITIES (COPIED FROM MEDQA_KL_BENCHMARK) =====

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
        uniform_dist = torch.ones(n_outputs, device=device) / n_outputs

        # Calculate KL divergences
        kl_argmax = kl_divergence(one_hot_expectation, uniform_dist).item()
        kl_prob = kl_divergence(prob_expectation, uniform_dist).item()

        kl_values_argmax.append(kl_argmax)
        kl_values_prob.append(kl_prob)

        # Calculate accuracy if labels are provided
        if labels is not None:
            predicted_labels = np.argmax(outputs[fraction], axis=1)
            if isinstance(labels, torch.Tensor):
                true_labels = labels.cpu().numpy()
            else:
                true_labels = np.array(labels)

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

        aggregated[method] = {
            'kl_baseline_mean_prob': np.mean(kl_prob_values),
            'kl_baseline_std_prob': np.std(kl_prob_values),
            'kl_baseline_mean_onehot': np.mean(kl_argmax_values),
            'kl_baseline_std_onehot': np.std(kl_argmax_values),
            'fraction_wise_results': fraction_wise_results
        }

        # Add accuracy if available
        if 'average_accuracy' in results_list[0]:
            accuracy_values = [r['average_accuracy'] for r in results_list]
            aggregated[method]['average_accuracy_mean'] = np.mean(accuracy_values)
            aggregated[method]['average_accuracy_std'] = np.std(accuracy_values)

    return aggregated

def save_results(results, save_dir="./results", model_info="qlora"):
    """Save QLoRA results to JSON and generate comparison table."""
    # Create directories
    os.makedirs(f"{save_dir}/json", exist_ok=True)

    # Save to JSON
    json_file = f"{save_dir}/json/qlora_results_medqa.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"QLoRA results saved to {json_file}")

    # Generate comparison table
    table_data = []
    headers = ["Method", "Average KL (Prob)", "Average KL (Argmax)", "Average Accuracy"]

    for method, data in results.items():
        kl_prob = data['kl_baseline_mean_prob']
        kl_prob_std = data['kl_baseline_std_prob']
        kl_argmax = data['kl_baseline_mean_onehot']
        kl_argmax_std = data['kl_baseline_std_onehot']

        row = [
            f"QLoRA ({model_info})",
            f"{kl_prob:.2e} ± {kl_prob_std:.2e}",
            f"{kl_argmax:.2e} ± {kl_argmax_std:.2e}"
        ]

        if 'average_accuracy_mean' in data:
            acc_mean = data['average_accuracy_mean']
            acc_std = data['average_accuracy_std']
            row.append(f"{acc_mean:.3f} ± {acc_std:.3f}")
        else:
            row.append("N/A")

        table_data.append(row)

    # Generate table
    table = tabulate(table_data, headers=headers, tablefmt="grid")

    # Save table
    table_file = f"{save_dir}/qlora_comparison_table_medqa.txt"
    n_runs = 1  # QLoRA typically run once due to training cost

    with open(table_file, 'w') as f:
        f.write(f"QLoRA KL Divergence Results for MedQA (model: {model_info}):\n")
        f.write(table)

    print(f"QLoRA comparison table saved to {table_file}")
    print(f"\nQLoRA KL Divergence Results for MedQA:")
    print(table)

    return json_file, table_file

def process_medqa_qlora_dataset(lora_adapter_path, device="cuda", save_dir="./results",
                               n_samples=10, n_fractions=10,
                               base_model_path="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                               use_real_data=True, balanced=True):
    """Process MedQA dataset with QLoRA model and calculate KL benchmarks."""

    print("="*60)
    print("MedQA QLoRA KL Divergence Benchmark")
    print("="*60)
    print(f"LoRA adapters: {lora_adapter_path}")
    print(f"Samples: {n_samples}")
    print(f"Fractions: {n_fractions}")
    print(f"Device: {device}")
    print()

    # Load QLoRA data and run benchmark
    predictions, labels = load_medqa_data_qlora(
        lora_adapter_path=lora_adapter_path,
        n_samples=n_samples,
        n_fractions=n_fractions,
        base_model_path=base_model_path,
        use_real_data=use_real_data,
        balanced=balanced
    )

    # Calculate KL metrics
    kl_results = calculate_kl_metrics(predictions.numpy(), labels.numpy(), torch.device(device))

    # Aggregate results (single run for QLoRA)
    all_results = {'qlora_binomial': [kl_results]}
    aggregated_results = aggregate_results(all_results)

    # Save results
    model_info = f"binomial_ablated_{Path(lora_adapter_path).name}"
    save_results(aggregated_results, save_dir, model_info)

    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description="MedQA QLoRA KL Divergence Benchmark")
    parser.add_argument("--lora_adapters", type=str, required=True,
                       help="Path to trained LoRA adapters")
    parser.add_argument("--base_model", type=str,
                       default="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                       help="Path to base LLaMA model")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples to test")
    parser.add_argument("--fractions", type=int, default=10,
                       help="Number of ablation fractions to test")
    parser.add_argument("--use_real_data", action="store_true", default=True,
                       help="Use real MedQA dataset")
    parser.add_argument("--balanced", action="store_true", default=True,
                       help="Use balanced answer distribution")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Run QLoRA benchmark
    aggregated_results = process_medqa_qlora_dataset(
        lora_adapter_path=args.lora_adapters,
        device=device,
        save_dir=args.save_dir,
        n_samples=args.samples,
        n_fractions=args.fractions,
        base_model_path=args.base_model,
        use_real_data=args.use_real_data,
        balanced=args.balanced
    )

    print("\nQLoRA benchmark completed! 🎉")

if __name__ == "__main__":
    main()