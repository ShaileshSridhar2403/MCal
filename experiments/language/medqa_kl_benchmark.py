#!/usr/bin/env python3
"""
MedQA KL Divergence Benchmark - MCal Implementation

Following the exact pattern of vision KL benchmarks (mri_kl_benchmark.py, etc.)
but for language models with MedQA dataset and LLaMA predictions.

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
sys.path.insert(0, str(mcal_root / "src"))

# Import MCal utilities
from src.utils.optimization import get_expectation, make_one_hot, kl_divergence

# Import calibrator modules
from src.calibrators.mcal import MCal
from src.calibrators.mcal_ce import MCal_CE
from src.calibrators.platt import PlattCalibrator
from src.calibrators.temperature import TemperatureScaling

# Import transform modules for backward compatibility
from src.transforms.lambda_transforms import ExpectationLambdaTransform, OptimizedLambdaTransform
from src.transforms.logits import LogitsSharpTransform

# Import our self-contained MedQA utilities
from medqa_utils import (
    MCal_LLaMAModel,
    load_local_medqa_data,
    load_real_medqa_data,
    # load_synthetic_medqa_data,
    generate_fractionwise_predictions,
    generate_fractionwise_predictions_with_token_dropping,
    generate_fractionwise_predictions_with_attention_mask,
    create_medqa_prompt,
    map_probs_to_list
)


def load_medqa_llama_model(model_path, device=None):
    """Load LLaMA model for MedQA predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        result['kl_values_accuracy'] = accuracy_values
        result['average_accuracy'] = avg_accuracy
    # pdb.set_trace()
    return result

def apply_transform(outputs, labels, method, device=None, **kwargs):
    """Apply a transformation method to outputs - IDENTICAL to vision benchmarks."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Applying {method} transform...")

    if method == 'baseline':
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

    elif method == 'expectation_prob':
        return apply_expectation_prob_transform(outputs, device)

    elif method == 'expectation_onehot':
        return apply_expectation_onehot_transform(outputs, device)

    elif method == 'optimized_lambda':
        return apply_optimized_lambda_transform(outputs, device, **kwargs)

    elif method == 'logits_sharp':
        return apply_logits_sharp_transform(outputs, device, **kwargs)

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

def apply_mcal_ce_calibrator(outputs_tensor, target_labels, device, max_steps=5000, head_type="linear", experiment_id="medqa_experiment", **kwargs):
    """Apply MCal_CE calibrator using cross-entropy loss."""
    # Convert numpy arrays to torch tensors if needed
    if not isinstance(outputs_tensor, torch.Tensor):
        outputs_tensor = torch.tensor(outputs_tensor, dtype=torch.float32, device=device)
    if not isinstance(target_labels, torch.Tensor):
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)
    
    target_labels = outputs_tensor[0].argmax(dim=-1)

    n_fractions, n_samples, n_classes = outputs_tensor.shape
    transformed_outputs = np.zeros_like(outputs_tensor.cpu().numpy())

    for fraction in tqdm(range(n_fractions), desc="Applying MCal_CE calibrator"):
        calibrator = MCal_CE(num_classes=n_classes, head_type="linear")
        calibrator.to(device)
        # pdb.set_trace()
        calibrator.fit(
            ablated_probs=outputs_tensor[fraction],
            target_labels=target_labels,
            max_steps=max_steps,
            lr=1e-2,
            verbose=True,
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


def apply_mcal_ce_uncond_calibrator(outputs_tensor, target_labels, device, max_steps=5000, head_type="linear", experiment_id="medqa_experiment", **kwargs):
    """Apply MCal_CE_Uncond calibrator (unconditional training approach)."""
    # Convert numpy arrays to torch tensors if needed
    if not isinstance(outputs_tensor, torch.Tensor):
        outputs_tensor = torch.tensor(outputs_tensor, dtype=torch.float32, device=device)
    if not isinstance(target_labels, torch.Tensor):
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

    # Use fraction 0 predictions as target labels (following MedQA pattern)
    target_labels = outputs_tensor[0].argmax(dim=-1)

    n_fractions, n_samples, n_classes = outputs_tensor.shape
    transformed_outputs = np.zeros_like(outputs_tensor.detach().cpu().numpy())
    train_tensor = torch.zeros_like(outputs_tensor[0])

    # Create training data by randomly sampling from all fractions
    for i in range(n_samples):
        fraction_ind = np.random.binomial(n_fractions-1, 0.5)  # Fix: n_fractions-1 to avoid out-of-bounds
        train_tensor[i, :] = outputs_tensor[fraction_ind][i]

    # Train single MCal_CE calibrator
    calibrator = MCal_CE(num_classes=n_classes, head_type=head_type)
    calibrator.to(device)
    calibrator.fit(
        ablated_probs=train_tensor,
        target_labels=target_labels,
        max_steps=max_steps,
        lr=1e-2,  # Match MedQA pattern
        verbose=True,
        fraction=0,  # Placeholder for unconditional training
        experiment_id=experiment_id
    )

    # Apply the single calibrator to all fractions
    for fraction in tqdm(range(n_fractions), desc="Applying MCal_CE_Uncond calibrator"):
        calibrated_probs = calibrator.forward(outputs_tensor[fraction])
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    # Combine results
    print(f"\n=== Combining MCal_CE_Uncond results for experiment: {experiment_id} ===")
    combined_file = MCal_CE.combine_fraction_results(experiment_id, cleanup_temp_files=True)
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

# Stub implementations for other transforms (can be implemented later)
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

def apply_logits_sharp_transform(outputs, device, **kwargs):
    """Stub - apply logits sharp transform."""
    print("⚠️  Logits sharp transform not implemented yet")
    return outputs

def aggregate_fractionwise_kl(fractionwise_results):
    """Aggregate fractionwise KL divergence results across multiple runs."""
    if not fractionwise_results or not fractionwise_results[0]:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": [], "mean_accuracy": [], "std_accuracy": []}

    first_result = fractionwise_results[0]
    if isinstance(first_result, dict) and 'kl_values_argmax' in first_result:
        num_fractions = len(first_result['kl_values_argmax'])
    else:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": [], "mean_accuracy": [], "std_accuracy": []}

    kl_argmax_values = [[] for _ in range(num_fractions)]
    kl_prob_values = [[] for _ in range(num_fractions)]
    accuracy_values = [[] for _ in range(num_fractions)]

    for run_results in fractionwise_results:
        kl_argmax_list = run_results['kl_values_argmax']
        kl_prob_list = run_results['kl_values_prob']
        accuracy_list = run_results.get('kl_values_accuracy', [])

        for i in range(min(len(kl_argmax_list), num_fractions)):
            kl_argmax_values[i].append(kl_argmax_list[i])
            kl_prob_values[i].append(kl_prob_list[i])

            # Add accuracy if available
            if i < len(accuracy_list):
                accuracy_values[i].append(accuracy_list[i])

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

def aggregate_results(all_results):
    """Aggregate results across multiple runs."""
    aggregated_results = {}

    for method, results in all_results.items():
        if not results:
            continue

        kl_prob_values = [r['average_kl_prob'] for r in results]
        kl_argmax_values = [r['average_kl_argmax'] for r in results]

        # Extract accuracy values if available
        accuracy_values = [r.get('average_accuracy', 0) for r in results if r and 'average_accuracy' in r]

        fraction_wise_results = aggregate_fractionwise_kl(results)

        aggregated_results[method] = {
            'kl_transformed_mean_prob': np.mean(kl_prob_values),
            'kl_transformed_std_prob': np.std(kl_prob_values),
            'kl_transformed_mean_onehot': np.mean(kl_argmax_values),
            'kl_transformed_std_onehot': np.std(kl_argmax_values),
            'fraction_wise_results_transformed': fraction_wise_results
        }

        # Add accuracy metrics if available
        if accuracy_values:
            aggregated_results[method]['accuracy_transformed_mean'] = np.mean(accuracy_values)
            aggregated_results[method]['accuracy_transformed_std'] = np.std(accuracy_values) if len(accuracy_values) > 1 else 0.0

        # For baseline, also store as baseline results
        if method == 'baseline':
            aggregated_results['baseline'] = {
                'kl_baseline_mean_prob': np.mean(kl_prob_values),
                'kl_baseline_std_prob': np.std(kl_prob_values),
                'kl_baseline_mean_onehot': np.mean(kl_argmax_values),
                'kl_baseline_std_onehot': np.std(kl_argmax_values),
                'fraction_wise_results': fraction_wise_results
            }

    return aggregated_results

def build_kl_comparison_table(aggregated_results, include_methods=None):
    """Build comparison table for KL divergence results."""
    table_data = [["Method", "Average KL (Prob)", "Average KL (Argmax)"]]

    method_names = {
        'baseline': "Original",
        'mcal': "MCal (Vector Scaling)",
        'mcal_ce': "MCal_CE (Cross-Entropy)",
        'mcal_ce_uncond': "MCal_CE_Uncond (Unconditional)",
        'platt': "Platt Scaling",
        'temperature': "Temperature Scaling",
        'token_drop': "Token Dropping",
        'logits_sharp': "Logits Sharp Transform",
        'expectation_prob': "Expectation Probability Transform",
        'expectation_onehot': "Expectation One-hot Transform",
        'optimized_lambda': "Optimized Lambda Transform"
    }

    # Add baseline
    if 'baseline' in aggregated_results:
        baseline = aggregated_results['baseline']
        table_data.append([
            method_names['baseline'],
            f"{baseline['kl_baseline_mean_prob']:.2e} ± {baseline['kl_baseline_std_prob']:.2e}",
            f"{baseline['kl_baseline_mean_onehot']:.2e} ± {baseline['kl_baseline_std_onehot']:.2e}"
        ])

    # Add other methods
    methods_to_include = include_methods or [m for m in aggregated_results.keys() if m != 'baseline']

    for method in methods_to_include:
        if method not in aggregated_results or method == 'baseline':
            continue

        result = aggregated_results[method]
        if 'kl_transformed_mean_prob' in result:
            table_data.append([
                method_names.get(method, method.replace('_', ' ').title()),
                f"{result['kl_transformed_mean_prob']:.2e} ± {result['kl_transformed_std_prob']:.2e}",
                f"{result['kl_transformed_mean_onehot']:.2e} ± {result['kl_transformed_std_onehot']:.2e}"
            ])

    table = tabulate(table_data, headers="firstrow", tablefmt="grid")
    return table

def load_medqa_data(model_type="vanilla", n_samples=10, n_fractions=10,
                   model_path="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                   use_real_data=True, balanced=True):
    """Load MedQA data following vision benchmark pattern."""

    print(f"Loading MedQA data with {n_samples} samples, {n_fractions} fractions...")
    print(f"Real data: {use_real_data}, Balanced: {balanced}")

    if model_type in ["vanilla", "token_drop", "attention_mask", "qlora"]:
        # Load model
        expanded_path = Path(model_path).expanduser()
        model = load_medqa_llama_model(str(expanded_path))

        # Load MedQA data (local, real, or synthetic)
        if use_real_data:
            # First try local balanced data, then fall back to online/synthetic
            medqa_questions = load_local_medqa_data(n_samples, balanced=balanced)
        else:
            print("Using synthetic MedQA data...")
            medqa_questions = load_synthetic_medqa_data(n_samples)

        # Generate predictions with different ablation fractions
        removal_fractions = np.linspace(0, 0.9, n_fractions).tolist()

        if model_type == "token_drop":
            # Use token dropping strategy
            predictions = generate_fractionwise_predictions_with_token_dropping(
                model=model,
                data=medqa_questions,
                removal_fractions=removal_fractions,
                prompt_type='default',
                batch_size=min(8, n_samples),
                num_options=5,
                use_tokenizer=True  # Enable token dropping
            )
        elif model_type == "attention_mask":
            # Use attention masking strategy (content-only)
            predictions = generate_fractionwise_predictions_with_attention_mask(
                model=model,
                data=medqa_questions,
                removal_fractions=removal_fractions,
                prompt_type='default',
                batch_size=min(8, n_samples),
                num_options=5
            )

        elif model_type == "qlora":
            # Use qlora strategy
            model = MCal_LLaMAModel("~/foo/MCal/saved_models/medqa/medqa_p0.5/merged_model")
            predictions = generate_fractionwise_predictions(
                model=model,
                data=medqa_questions,
                removal_fractions=removal_fractions,
                prompt_type='default',
                batch_size=min(8, n_samples),
                num_options=5,
            )
        else:
            # Use standard word replacement strategy
            predictions = generate_fractionwise_predictions(
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
            # Fallback to argmax of clean predictions for synthetic data
            clean_predictions = predictions[0]  # First fraction (no ablation)
            labels = np.argmax(clean_predictions, axis=1)
            print(f"Using argmax of clean predictions as labels")

        print(f"Generated predictions shape: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")

        # Convert to torch tensors
        predictions = torch.from_numpy(predictions).float()
        labels = torch.from_numpy(labels).long()

        # pdb.set_trace()
        model.model.cpu()
        del model

        return predictions, labels

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def process_medqa_dataset(methods=None, device="cuda", save_dir="./results", n_runs=3,
                         n_samples=10, n_fractions=10,
                         model_path="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                         use_real_data=True, balanced=True):
    """Process MedQA dataset and generate KL benchmarks - IDENTICAL STRUCTURE to vision."""

    if methods is None:
        methods = ['baseline', 'mcal', 'mcal_ce', 'mcal_ce_uncond', 'platt', 'temperature']

    device = torch.device(device)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)

    print("="*60)
    print("MedQA KL Divergence Benchmark")
    print("="*60)
    print(f"Methods: {methods}")
    print(f"Runs: {n_runs}")
    print(f"Samples per run: {n_samples}")
    print(f"Fractions: {n_fractions}")
    print(f"Device: {device}")

    # Initialize results storage
    all_results = {method: [] for method in methods}

    # Run multiple experiments
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Check which data types we need
        need_token_drop = 'token_drop' in methods
        need_attention_mask = 'attention_mask' in methods
        need_qlora = 'qlora' in methods
        need_vanilla = any(method not in ['token_drop', 'attention_mask', "qlora"] for method in methods)

        # Load data for different ablation strategies
        all_predictions = {}
        all_labels = {}

        if need_vanilla:
            # Load vanilla (word replacement) data
            predictions_vanilla, labels_vanilla = load_medqa_data(
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
            predictions_token_drop, labels_token_drop = load_medqa_data(
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
            predictions_attention_mask, labels_attention_mask = load_medqa_data(
                model_type="attention_mask",
                n_samples=n_samples,
                n_fractions=n_fractions,
                model_path=model_path,
                use_real_data=use_real_data,
                balanced=balanced
            )
            all_predictions['attention_mask'] = predictions_attention_mask
            all_labels['attention_mask'] = labels_attention_mask

        if need_qlora:
            # Load qlora data
            predictions_qlora, labels_qlora = load_medqa_data(
                model_type="qlora",
                n_samples=n_samples,
                n_fractions=n_fractions,
            )
            all_predictions['qlora'] = predictions_qlora
            all_labels['qlora'] = labels_qlora

        # Process each method
        for method in methods:
            print(f"\nProcessing method: {method}")

            # Select appropriate predictions and labels
            if method == 'token_drop':
                predictions = all_predictions['token_drop']
                labels = all_labels['token_drop']
                # For token_drop, the baseline predictions are already the "transformed" ones
                transformed_predictions = predictions
            elif method == 'attention_mask':
                predictions = all_predictions['attention_mask']
                labels = all_labels['attention_mask']
                # For attention_mask, the baseline predictions are already the "transformed" ones
                transformed_predictions = predictions
            elif method == 'qlora':
                predictions = all_predictions['qlora']
                labels = all_labels['qlora']
                # For qlora, the baseline predictions are already the "transformed" ones
                transformed_predictions = predictions
            else:
                predictions = all_predictions['vanilla']
                labels = all_labels['vanilla']

                # Apply transformation
                if method == 'baseline':
                    transformed_predictions = predictions
                else:
                    # Configure method-specific parameters
                    method_kwargs = {}
                    if method in ['mcal', 'mcal_ce', 'mcal_ce_uncond', 'platt', 'temperature']:
                        method_kwargs['max_steps'] = 1000
                    if method == 'mcal':
                        method_kwargs['kappa'] = 10.0
                    elif method in ['mcal_ce', 'mcal_ce_uncond']:
                        method_kwargs['max_steps'] = 5000
                        method_kwargs['head_type'] = 'linear'

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
    # pdb.set_trace()
    # Save results as JSON
    json_path = os.path.join(save_dir, "json", "aggregated_results_medqa.json")
    json_serializable_results = convert_to_json_serializable(aggregated_results)

    with open(json_path, 'w') as f:
        json.dump(json_serializable_results, f, indent=4)
    print(f"Aggregated results saved to {json_path}")

    # Build and display comparison table
    table = build_kl_comparison_table(aggregated_results, include_methods=methods)

    print(f"\nKL Divergence Comparison for MedQA (averaged over {n_runs} runs):")
    print(table)

    # Save table
    table_path = os.path.join(save_dir, "kl_comparison_table_medqa.txt")
    with open(table_path, 'w') as f:
        f.write(f"KL Divergence Comparison for MedQA (averaged over {n_runs} runs):\n")
        f.write(table)
    print(f"Comparison table saved to {table_path}")

    # Display fractionwise accuracy summary (following MRI pattern)
    print(f"\nFractionwise Accuracy Summary:")
    for method, result in aggregated_results.items():
        if method == 'baseline' and 'fraction_wise_results' in result:
            fwr = result['fraction_wise_results']
            if 'mean_accuracy' in fwr and any(acc > 0 for acc in fwr['mean_accuracy']):
                avg_accuracy = np.mean(fwr['mean_accuracy'])
                std_accuracy = np.std(fwr['mean_accuracy'])
                print(f"  {method.upper()}: Average fractionwise accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
                if 'accuracy_transformed_mean' in result:
                    overall_mean = result['accuracy_transformed_mean']
                    overall_std = result.get('accuracy_transformed_std', 0.0)
                    print(f"    Overall accuracy: {overall_mean:.4f} ± {overall_std:.4f}")
        elif 'fraction_wise_results_transformed' in result:
            fwr = result['fraction_wise_results_transformed']
            if 'mean_accuracy' in fwr and any(acc > 0 for acc in fwr['mean_accuracy']):
                avg_accuracy = np.mean(fwr['mean_accuracy'])
                std_accuracy = np.std(fwr['mean_accuracy'])
                print(f"  {method.upper()}: Average fractionwise accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
                if 'accuracy_transformed_mean' in result:
                    overall_mean = result['accuracy_transformed_mean']
                    overall_std = result.get('accuracy_transformed_std', 0.0)
                    print(f"    Overall accuracy: {overall_mean:.4f} ± {overall_std:.4f}")

    return aggregated_results

def convert_to_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON serializable types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MedQA KL Divergence Benchmark")
    parser.add_argument("--methods", nargs='+',
                       default=['baseline', 'mcal_ce', 'platt', 'temperature', "qlora", "attention_mask", "token_drop"],
                       help="Methods to include in benchmark (baseline, mcal, mcal_ce, platt, temperature, token_drop, attention_mask)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per run")
    parser.add_argument("--fractions", type=int, default=10, help="Number of fractions")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="./results", help="Save directory")
    parser.add_argument("--model_path", type=str,
                       default="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                       help="Path to LLaMA model")
    parser.add_argument("--use_real_data", action="store_true", default=True,
                       help="Use real MedQA dataset (default: True)")
    parser.add_argument("--use_synthetic_data", action="store_true", default=False,
                       help="Use synthetic MedQA data instead of real data")
    parser.add_argument("--balanced", action="store_true", default=True,
                       help="Ensure balanced answer distribution (default: True)")

    args = parser.parse_args()

    # Handle data source flags
    use_real_data = args.use_real_data and not args.use_synthetic_data

    # Set device
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {device}")

    # Run benchmark
    aggregated_results = process_medqa_dataset(
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