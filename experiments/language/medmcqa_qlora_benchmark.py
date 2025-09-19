#!/usr/bin/env python3
"""
MedMCQA QLoRA KL Divergence Benchmark

Benchmarks QLoRA models trained on binomially ablated data against various ablation methods.
Measures KL divergence from uniform distribution and applies MCal calibration methods.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import json
from tqdm import tqdm
import logging

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))

from medmcqa_utils import (
    MCal_LLaMAModel,
    load_medmcqa_data,
    create_medmcqa_prompt,
    replace_random_features,
    remove_random_tokens_with_tokenizer,
    create_random_attention_mask,
    identify_content_positions
)
from qlora_utils import MCal_QLoRA_Model
from src.KL_divergence import KL_divergence
from src.MCal import MCal
from src.temperature_scaling import temperature_scaling
from src.platt_scaling import platt_scaling
from src.utilities.plotting import MCal_Result

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_fractionwise_predictions_qlora(model, data, removal_fractions, prompt_type='default', batch_size=8, num_options=4):
    """Generate predictions using QLoRA model for different ablation fractions."""
    logger.info(f"Generating QLoRA predictions for {len(data)} samples with fractions: {removal_fractions}")

    all_predictions = {}

    for fraction in tqdm(removal_fractions, desc="Processing ablation fractions"):
        logger.info(f"Processing fraction: {fraction}")
        predictions = []
        true_labels = []

        for i in tqdm(range(0, len(data), batch_size), desc=f"Processing batches for fraction {fraction}", leave=False):
            batch_data = data[i:i+batch_size]

            for question_data in batch_data:
                # Create prompt
                prompt = create_medmcqa_prompt(question_data, prompt_type=prompt_type)

                # Get prediction from QLoRA model (which handles its own ablation)
                prediction = model.get_choice_probabilities(prompt, num_options=num_options)
                predictions.append(prediction)

                # Convert 1-indexed answer to 0-indexed
                true_label = question_data['cop'] - 1
                true_labels.append(true_label)

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        all_predictions[fraction] = {
            'predictions': predictions,
            'true_labels': true_labels
        }

        # Calculate and log accuracy
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        mean_probs = np.mean(predictions, axis=0)
        logger.info(f"Fraction {fraction:.2f} - Accuracy: {accuracy:.3f}, Mean probs: {mean_probs}")

    return all_predictions

def generate_fractionwise_predictions_with_word_replacement(model, data, removal_fractions, prompt_type='default', batch_size=8, num_options=4):
    """Generate predictions using word replacement for different ablation fractions."""
    logger.info(f"Generating word replacement predictions for {len(data)} samples with fractions: {removal_fractions}")

    all_predictions = {}

    for fraction in tqdm(removal_fractions, desc="Processing ablation fractions"):
        logger.info(f"Processing fraction: {fraction}")
        predictions = []
        true_labels = []

        for i in tqdm(range(0, len(data), batch_size), desc=f"Processing batches for fraction {fraction}", leave=False):
            batch_data = data[i:i+batch_size]

            for question_data in batch_data:
                # Create prompt
                prompt = create_medmcqa_prompt(question_data, prompt_type=prompt_type)

                # Apply word replacement ablation
                ablated_prompt = replace_random_features(prompt, fraction, ablation_method='word', seed=None)

                # Get prediction
                prediction = model.get_choice_probabilities(ablated_prompt, num_options=num_options)
                predictions.append(prediction)

                # Convert 1-indexed answer to 0-indexed
                true_label = question_data['cop'] - 1
                true_labels.append(true_label)

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        all_predictions[fraction] = {
            'predictions': predictions,
            'true_labels': true_labels
        }

        # Calculate and log accuracy
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        mean_probs = np.mean(predictions, axis=0)
        logger.info(f"Fraction {fraction:.2f} - Accuracy: {accuracy:.3f}, Mean probs: {mean_probs}")

    return all_predictions

def generate_fractionwise_predictions_with_token_dropping(model, data, removal_fractions, prompt_type='default', batch_size=8, num_options=4):
    """Generate predictions using token dropping for different ablation fractions."""
    logger.info(f"Generating token dropping predictions for {len(data)} samples with fractions: {removal_fractions}")

    all_predictions = {}

    for fraction in tqdm(removal_fractions, desc="Processing ablation fractions"):
        logger.info(f"Processing fraction: {fraction}")
        predictions = []
        true_labels = []

        for i in tqdm(range(0, len(data), batch_size), desc=f"Processing batches for fraction {fraction}", leave=False):
            batch_data = data[i:i+batch_size]

            for question_data in batch_data:
                # Create prompt
                prompt = create_medmcqa_prompt(question_data, prompt_type=prompt_type)

                # Apply token dropping ablation
                ablated_prompt = remove_random_tokens_with_tokenizer(model.tokenizer, prompt, fraction, seed=None)

                # Get prediction
                prediction = model.get_choice_probabilities(ablated_prompt, num_options=num_options)
                predictions.append(prediction)

                # Convert 1-indexed answer to 0-indexed
                true_label = question_data['cop'] - 1
                true_labels.append(true_label)

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        all_predictions[fraction] = {
            'predictions': predictions,
            'true_labels': true_labels
        }

        # Calculate and log accuracy
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        mean_probs = np.mean(predictions, axis=0)
        logger.info(f"Fraction {fraction:.2f} - Accuracy: {accuracy:.3f}, Mean probs: {mean_probs}")

    return all_predictions

def generate_fractionwise_predictions_with_attention_mask(model, data, removal_fractions, prompt_type='default', batch_size=8, num_options=4):
    """Generate predictions using attention masking for different ablation fractions."""
    logger.info(f"Generating attention mask predictions for {len(data)} samples with fractions: {removal_fractions}")

    all_predictions = {}

    for fraction in tqdm(removal_fractions, desc="Processing ablation fractions"):
        logger.info(f"Processing fraction: {fraction}")
        predictions = []
        true_labels = []

        for i in tqdm(range(0, len(data), batch_size), desc=f"Processing batches for fraction {fraction}", leave=False):
            batch_data = data[i:i+batch_size]

            for question_data in batch_data:
                # Create prompt
                prompt = create_medmcqa_prompt(question_data, prompt_type=prompt_type)

                # Create attention mask
                input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
                content_positions = identify_content_positions(model.tokenizer, prompt, question_data)
                attention_mask = create_random_attention_mask(input_ids, content_positions, fraction, seed=None)

                # Get prediction with attention mask
                prediction = model.get_choice_probabilities_with_attention_mask(prompt, attention_mask, num_options=num_options)
                predictions.append(prediction)

                # Convert 1-indexed answer to 0-indexed
                true_label = question_data['cop'] - 1
                true_labels.append(true_label)

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        all_predictions[fraction] = {
            'predictions': predictions,
            'true_labels': true_labels
        }

        # Calculate and log accuracy
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        mean_probs = np.mean(predictions, axis=0)
        logger.info(f"Fraction {fraction:.2f} - Accuracy: {accuracy:.3f}, Mean probs: {mean_probs}")

    return all_predictions

def apply_mcal_calibrator(outputs, device, kappa=4.0, max_steps=10000, **kwargs):
    """Apply MCal calibration."""
    n_classes = outputs.shape[1]
    uniform_target = np.ones(n_classes) / n_classes

    calibrator = MCal(num_classes=n_classes, target_distribution=uniform_target)
    calibrator.kappa = kappa
    calibrator.max_steps = max_steps

    # Convert to tensors if needed
    if isinstance(outputs, np.ndarray):
        outputs_tensor = torch.from_numpy(outputs).float().to(device)
    else:
        outputs_tensor = outputs.to(device)

    calibrated_outputs = calibrator.calibrate(outputs_tensor)

    if isinstance(calibrated_outputs, torch.Tensor):
        calibrated_outputs = calibrated_outputs.cpu().numpy()

    return calibrated_outputs

def apply_mcal_ce_calibrator(outputs, true_labels, device, kappa=4.0, max_steps=10000, **kwargs):
    """Apply MCal CE calibration."""
    from src.MCal_CE import MCal_CE

    n_classes = outputs.shape[1]
    uniform_target = np.ones(n_classes) / n_classes

    calibrator = MCal_CE(num_classes=n_classes, target_distribution=uniform_target)
    calibrator.kappa = kappa
    calibrator.max_steps = max_steps

    # Convert to tensors if needed
    if isinstance(outputs, np.ndarray):
        outputs_tensor = torch.from_numpy(outputs).float().to(device)
    else:
        outputs_tensor = outputs.to(device)

    if isinstance(true_labels, np.ndarray):
        labels_tensor = torch.from_numpy(true_labels).long().to(device)
    else:
        labels_tensor = true_labels.to(device)

    calibrated_outputs = calibrator.calibrate(outputs_tensor, labels_tensor)

    if isinstance(calibrated_outputs, torch.Tensor):
        calibrated_outputs = calibrated_outputs.cpu().numpy()

    return calibrated_outputs

def apply_temperature_scaling(outputs, true_labels, device, **kwargs):
    """Apply temperature scaling calibration."""
    if isinstance(outputs, np.ndarray):
        outputs_tensor = torch.from_numpy(outputs).float().to(device)
    else:
        outputs_tensor = outputs.to(device)

    if isinstance(true_labels, np.ndarray):
        labels_tensor = torch.from_numpy(true_labels).long().to(device)
    else:
        labels_tensor = true_labels.to(device)

    calibrated_outputs = temperature_scaling(outputs_tensor, labels_tensor)

    if isinstance(calibrated_outputs, torch.Tensor):
        calibrated_outputs = calibrated_outputs.cpu().numpy()

    return calibrated_outputs

def apply_platt_scaling(outputs, true_labels, device, **kwargs):
    """Apply Platt scaling calibration."""
    if isinstance(outputs, np.ndarray):
        outputs_tensor = torch.from_numpy(outputs).float().to(device)
    else:
        outputs_tensor = outputs.to(device)

    if isinstance(true_labels, np.ndarray):
        labels_tensor = torch.from_numpy(true_labels).long().to(device)
    else:
        labels_tensor = true_labels.to(device)

    calibrated_outputs = platt_scaling(outputs_tensor, labels_tensor)

    if isinstance(calibrated_outputs, torch.Tensor):
        calibrated_outputs = calibrated_outputs.cpu().numpy()

    return calibrated_outputs

def run_kl_benchmark(model, medmcqa_questions, removal_fractions, calibration_methods, model_type, device, num_options=4):
    """Run KL divergence benchmark with specified model and ablation method."""
    logger.info(f"Running KL benchmark with {model_type} model")

    # Generate predictions based on model type
    if model_type == "qlora":
        predictions = generate_fractionwise_predictions_qlora(
            model=model, data=medmcqa_questions, removal_fractions=removal_fractions,
            prompt_type='default', batch_size=min(8, len(medmcqa_questions)), num_options=num_options
        )
    elif model_type == "word_replacement":
        predictions = generate_fractionwise_predictions_with_word_replacement(
            model=model, data=medmcqa_questions, removal_fractions=removal_fractions,
            prompt_type='default', batch_size=min(8, len(medmcqa_questions)), num_options=num_options
        )
    elif model_type == "token_drop":
        predictions = generate_fractionwise_predictions_with_token_dropping(
            model=model, data=medmcqa_questions, removal_fractions=removal_fractions,
            prompt_type='default', batch_size=min(8, len(medmcqa_questions)), num_options=num_options
        )
    elif model_type == "attention_mask":
        predictions = generate_fractionwise_predictions_with_attention_mask(
            model=model, data=medmcqa_questions, removal_fractions=removal_fractions,
            prompt_type='default', batch_size=min(8, len(medmcqa_questions)), num_options=num_options
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Prepare results storage
    fraction_wise_results = {}

    for method in calibration_methods:
        logger.info(f"Applying calibration method: {method}")
        fraction_wise_results[method] = {
            'kl_divergences': [],
            'accuracies': []
        }

        for fraction in removal_fractions:
            outputs = predictions[fraction]['predictions']
            true_labels = predictions[fraction]['true_labels']

            # Apply calibration method
            if method == 'none':
                calibrated_outputs = outputs
            elif method == 'MCal':
                calibrated_outputs = apply_mcal_calibrator(outputs, device)
            elif method == 'MCal_CE':
                calibrated_outputs = apply_mcal_ce_calibrator(outputs, true_labels, device)
            elif method == 'temperature_scaling':
                calibrated_outputs = apply_temperature_scaling(outputs, true_labels, device)
            elif method == 'platt_scaling':
                calibrated_outputs = apply_platt_scaling(outputs, true_labels, device)
            else:
                raise ValueError(f"Unknown calibration method: {method}")

            # Calculate KL divergence from uniform distribution
            uniform_dist = np.ones(num_options) / num_options
            kl_div = KL_divergence(calibrated_outputs, uniform_dist)

            # Calculate accuracy
            predicted_labels = np.argmax(calibrated_outputs, axis=1)
            accuracy = np.mean(predicted_labels == true_labels)

            fraction_wise_results[method]['kl_divergences'].append(kl_div)
            fraction_wise_results[method]['accuracies'].append(accuracy)

            logger.info(f"Method: {method}, Fraction: {fraction:.2f}, KL Div: {kl_div:.4f}, Accuracy: {accuracy:.3f}")

    return fraction_wise_results

def create_results_table(fraction_wise_results, removal_fractions, calibration_methods):
    """Create formatted results table."""
    logger.info("Creating results table")

    # Prepare data for MCal_Result
    data_dict = {}

    for method in calibration_methods:
        kl_values = fraction_wise_results[method]['kl_divergences']
        accuracy_values = fraction_wise_results[method]['accuracies']

        # Calculate summary statistics
        avg_kl = np.mean(kl_values)
        avg_accuracy = np.mean(accuracy_values)

        data_dict[f'{method}_kl'] = kl_values
        data_dict[f'{method}_accuracy'] = accuracy_values
        data_dict[f'{method}_avg_kl'] = [avg_kl] * len(removal_fractions)
        data_dict[f'{method}_avg_accuracy'] = [avg_accuracy] * len(removal_fractions)

    # Add removal fractions
    data_dict['removal_fraction'] = removal_fractions

    # Create MCal_Result
    mcal_result = MCal_Result(data_dict)

    return mcal_result

def main():
    parser = argparse.ArgumentParser(description="MedMCQA QLoRA KL Divergence Benchmark")

    # Model arguments
    parser.add_argument("--base_model", type=str,
                       default="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                       help="Path to base LLaMA model")
    parser.add_argument("--qlora_model", type=str,
                       help="Path to QLoRA model (optional)")

    # Dataset arguments
    parser.add_argument("--n_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--balanced", action="store_true", default=True,
                       help="Use balanced dataset")

    # Benchmark arguments
    parser.add_argument("--methods", nargs='+',
                       choices=["qlora", "word_replacement", "token_drop", "attention_mask"],
                       default=["qlora", "word_replacement"],
                       help="Ablation methods to compare")
    parser.add_argument("--calibration_methods", nargs='+',
                       choices=["none", "MCal", "MCal_CE", "temperature_scaling", "platt_scaling"],
                       default=["none", "MCal", "MCal_CE"],
                       help="Calibration methods to apply")
    parser.add_argument("--num_fractions", type=int, default=10,
                       help="Number of ablation fractions to test")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./medmcqa_qlora_results",
                       help="Directory to save results")

    args = parser.parse_args()

    # Expand paths
    args.base_model = str(Path(args.base_model).expanduser())
    if args.qlora_model:
        args.qlora_model = str(Path(args.qlora_model).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Starting MedMCQA QLoRA KL Divergence Benchmark")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Calibration methods: {args.calibration_methods}")
    logger.info(f"Samples: {args.n_samples}")

    # Load MedMCQA data
    logger.info("Loading MedMCQA data...")
    medmcqa_questions = load_medmcqa_data(
        n_samples=args.n_samples,
        balanced=args.balanced
    )
    logger.info(f"Loaded {len(medmcqa_questions)} questions")

    # Setup removal fractions
    removal_fractions = np.linspace(0.0, 0.9, args.num_fractions)
    logger.info(f"Removal fractions: {removal_fractions}")

    # Results storage
    all_results = {}

    # Run benchmarks for each method
    for method in args.methods:
        logger.info(f"Running benchmark for method: {method}")

        if method == "qlora":
            if not args.qlora_model:
                raise ValueError("QLoRA model path required for qlora method")
            model = MCal_QLoRA_Model(args.base_model, args.qlora_model)
        else:
            model = MCal_LLaMAModel(args.base_model)

        # Run KL benchmark
        results = run_kl_benchmark(
            model=model,
            medmcqa_questions=medmcqa_questions,
            removal_fractions=removal_fractions,
            calibration_methods=args.calibration_methods,
            model_type=method,
            device=device,
            num_options=4
        )

        all_results[method] = results

        # Clean up model to save memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create and save results tables
    for method in args.methods:
        logger.info(f"Creating results table for {method}")

        results_table = create_results_table(
            all_results[method],
            removal_fractions,
            args.calibration_methods
        )

        # Save results
        output_file = Path(args.output_dir) / f"medmcqa_{method}_kl_results.pkl"
        results_table.save(str(output_file))
        logger.info(f"Results saved to: {output_file}")

        # Save raw results as JSON
        json_file = Path(args.output_dir) / f"medmcqa_{method}_raw_results.json"
        with open(json_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for cal_method in all_results[method]:
                json_results[cal_method] = {
                    'kl_divergences': [float(x) for x in all_results[method][cal_method]['kl_divergences']],
                    'accuracies': [float(x) for x in all_results[method][cal_method]['accuracies']]
                }
            json.dump({
                'method': method,
                'calibration_methods': args.calibration_methods,
                'removal_fractions': [float(x) for x in removal_fractions],
                'results': json_results,
                'n_samples': args.n_samples
            }, f, indent=2)
        logger.info(f"Raw results saved to: {json_file}")

    logger.info("MedMCQA QLoRA benchmark completed successfully!")
    logger.info(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()