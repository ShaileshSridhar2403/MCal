#!/usr/bin/env python3
"""
MedQA Token Dropping Benchmark - MCal Implementation

Compares word replacement vs token dropping strategies for text ablation.
Follows XAI-Benchmark patterns but self-contained for MCal.
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
sys.path.insert(0, str(mcal_root))

# Import our self-contained MedQA utilities
from medqa_utils import (
    MCal_LLaMAModel,
    load_local_medqa_data,
    generate_fractionwise_predictions_with_token_dropping,
    map_probs_to_list
)

def calculate_kl_divergence(outputs, uniform_dist=None):
    """Calculate KL divergence from uniform distribution."""
    if uniform_dist is None:
        n_classes = outputs.shape[-1]
        uniform_dist = np.ones(n_classes) / n_classes

    epsilon = 1e-10
    outputs_norm = outputs / (np.sum(outputs, axis=-1, keepdims=True) + epsilon)
    outputs_norm = outputs_norm + epsilon
    uniform_dist_norm = uniform_dist + epsilon

    # KL divergence for probability distributions
    kl_prob = np.mean(np.sum(outputs_norm * np.log(outputs_norm / uniform_dist_norm), axis=-1))

    # KL divergence for one-hot (argmax) distributions
    argmax_indices = np.argmax(outputs_norm, axis=-1)
    one_hot = np.zeros_like(outputs_norm)
    one_hot[np.arange(len(one_hot)), argmax_indices] = 1.0
    one_hot = one_hot + epsilon
    kl_argmax = np.mean(np.sum(one_hot * np.log(one_hot / uniform_dist_norm), axis=-1))

    return kl_prob, kl_argmax

def load_medqa_llama_model(model_path):
    """Load LLaMA model for MedQA."""
    print(f"Loading LLaMA model from: {model_path}")
    model = MCal_LLaMAModel(model_path)
    return model

def load_medqa_data(n_samples, balanced=True, use_real_data=True):
    """Load MedQA dataset."""
    print(f"Loading MedQA data with {n_samples} samples...")
    print(f"Real data: {use_real_data}, Balanced: {balanced}")

    if use_real_data:
        medqa_questions = load_local_medqa_data(n_samples, balanced=balanced)
    else:
        raise ValueError("Synthetic data generation not implemented")

    return medqa_questions

def compare_ablation_strategies(model, data, removal_fractions, prompt_type='default', batch_size=8, num_options=5):
    """
    Compare word replacement vs token dropping strategies.

    Returns:
        dict: Results for both strategies
    """
    results = {}

    # Test word replacement strategy
    print("\n=== Testing Word Replacement Strategy ===")
    word_predictions = generate_fractionwise_predictions_with_token_dropping(
        model=model,
        data=data,
        removal_fractions=removal_fractions,
        prompt_type=prompt_type,
        batch_size=batch_size,
        num_options=num_options,
        use_tokenizer=False  # Word replacement
    )

    # Calculate KL divergences for word replacement
    word_kl_results = []
    for fraction_idx, fraction in enumerate(removal_fractions):
        kl_prob, kl_argmax = calculate_kl_divergence(word_predictions[fraction_idx])
        word_kl_results.append({
            'fraction': fraction,
            'kl_prob': kl_prob,
            'kl_argmax': kl_argmax
        })
        print(f"Word replacement - Fraction {fraction:.1f}: KL(prob)={kl_prob:.6f}, KL(argmax)={kl_argmax:.6f}")

    results['word_replacement'] = {
        'predictions': word_predictions,
        'kl_results': word_kl_results,
        'average_kl_prob': np.mean([r['kl_prob'] for r in word_kl_results]),
        'average_kl_argmax': np.mean([r['kl_argmax'] for r in word_kl_results])
    }

    # Test token dropping strategy
    print("\n=== Testing Token Dropping Strategy ===")
    token_predictions = generate_fractionwise_predictions_with_token_dropping(
        model=model,
        data=data,
        removal_fractions=removal_fractions,
        prompt_type=prompt_type,
        batch_size=batch_size,
        num_options=num_options,
        use_tokenizer=True  # Token dropping
    )

    # Calculate KL divergences for token dropping
    token_kl_results = []
    for fraction_idx, fraction in enumerate(removal_fractions):
        kl_prob, kl_argmax = calculate_kl_divergence(token_predictions[fraction_idx])
        token_kl_results.append({
            'fraction': fraction,
            'kl_prob': kl_prob,
            'kl_argmax': kl_argmax
        })
        print(f"Token dropping - Fraction {fraction:.1f}: KL(prob)={kl_prob:.6f}, KL(argmax)={kl_argmax:.6f}")

    results['token_dropping'] = {
        'predictions': token_predictions,
        'kl_results': token_kl_results,
        'average_kl_prob': np.mean([r['kl_prob'] for r in token_kl_results]),
        'average_kl_argmax': np.mean([r['kl_argmax'] for r in token_kl_results])
    }

    return results

def create_comparison_table(results):
    """Create a comparison table for different ablation strategies."""
    table_data = [
        ["Strategy", "Average KL (Prob)", "Average KL (Argmax)"]
    ]

    for strategy_name, strategy_results in results.items():
        strategy_display = strategy_name.replace('_', ' ').title()
        table_data.append([
            strategy_display,
            f"{strategy_results['average_kl_prob']:.6f}",
            f"{strategy_results['average_kl_argmax']:.6f}"
        ])

    return tabulate(table_data, headers="firstrow", tablefmt="grid")

def save_results(results, output_dir="./results"):
    """Save results to JSON and create comparison table."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare results for JSON serialization (convert numpy arrays)
    json_results = {}
    for strategy_name, strategy_results in results.items():
        json_results[strategy_name] = {
            'average_kl_prob': float(strategy_results['average_kl_prob']),
            'average_kl_argmax': float(strategy_results['average_kl_argmax']),
            'kl_results': [
                {
                    'fraction': float(r['fraction']),
                    'kl_prob': float(r['kl_prob']),
                    'kl_argmax': float(r['kl_argmax'])
                }
                for r in strategy_results['kl_results']
            ]
        }

    # Save JSON results
    json_file = os.path.join(output_dir, 'medqa_token_dropping_comparison.json')
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {json_file}")

    # Create and save comparison table
    table = create_comparison_table(results)
    table_file = os.path.join(output_dir, 'medqa_token_dropping_table.txt')
    with open(table_file, 'w') as f:
        f.write("MedQA Token Dropping vs Word Replacement Comparison:\n")
        f.write(table)
    print(f"Comparison table saved to {table_file}")

    return table

def main():
    parser = argparse.ArgumentParser(description='MedQA Token Dropping Benchmark')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to use')
    parser.add_argument('--fractions', type=int, default=5, help='Number of removal fractions to test')
    parser.add_argument('--model-path', type=str,
                       default='/home/antonxue/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct',
                       help='Path to LLaMA model')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--prompt-type', type=str, default='default',
                       choices=['default', 'COT', 'Debiasing'], help='Type of prompt to use')

    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 60)
    print("MedQA Token Dropping vs Word Replacement Benchmark")
    print("=" * 60)
    print(f"Samples: {args.samples}")
    print(f"Fractions: {args.fractions}")
    print(f"Batch size: {args.batch_size}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Device: {device}")
    print()

    # Generate removal fractions
    removal_fractions = np.linspace(0.0, 0.9, args.fractions)
    print(f"Testing removal fractions: {removal_fractions}")

    # Load model
    model = load_medqa_llama_model(args.model_path)

    # Load data
    medqa_questions = load_medqa_data(args.samples, balanced=True, use_real_data=True)

    # Run comparison
    results = compare_ablation_strategies(
        model=model,
        data=medqa_questions,
        removal_fractions=removal_fractions,
        prompt_type=args.prompt_type,
        batch_size=args.batch_size,
        num_options=5
    )

    # Print and save results
    print("\n" + "=" * 60)
    print("FINAL COMPARISON RESULTS")
    print("=" * 60)
    table = save_results(results)
    print(table)

    print("\nBenchmark completed! 🎉")

if __name__ == "__main__":
    main()