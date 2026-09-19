#!/usr/bin/env python3
"""
Hyperparameter tuning script for Temperature Scaling calibration.

This script performs grid search and random search over key hyperparameters
for temperature scaling to optimize calibration performance.
"""

import os
import sys
import json
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from scipy.stats import uniform, loguniform
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent

from mcal.calibrators.temperature import TemperatureScaling


def compute_calibration_metrics(probs, labels):
    """Compute calibration metrics including Expected Calibration Error (ECE)."""
    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    # ECE computation
    n_bins = 10
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

    # Negative log likelihood
    nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-8).mean()

    # Accuracy
    accuracy = accuracies.mean()

    return {
        'ece': ece,
        'nll': nll,
        'accuracy': accuracy
    }


class TemperatureScalingTuner:
    """Temperature scaling hyperparameter tuner."""

    def __init__(self, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.results = []

    def evaluate_config(self, train_probs, train_labels, val_probs, val_labels, config):
        """Evaluate a single hyperparameter configuration."""
        try:
            # Create model with specific config
            model = TemperatureScaling(self.num_classes)
            model.to(self.device)

            # Initialize temperature if specified
            if 'temp_init' in config:
                with torch.no_grad():
                    model.temperature.fill_(config['temp_init'])

            # Override fit method with custom optimizer if needed
            if config.get('optimizer', 'lbfgs') != 'lbfgs':
                stats = self._fit_custom_optimizer(
                    model, train_probs, train_labels, config
                )
            else:
                stats = model.fit(
                    ablated_probs=train_probs,
                    labels=train_labels,
                    max_steps=config['max_steps'],
                    lr=config['lr'],
                    verbose=False
                )

            # Evaluate on validation set
            with torch.no_grad():
                calibrated_probs = model.forward(val_probs)
                metrics = compute_calibration_metrics(calibrated_probs, val_labels)

            result = {
                'config': config,
                'final_temperature': model.temperature.item(),
                'training_steps': len(stats['loss']),
                'final_loss': stats['loss'][-1] if stats['loss'] else float('inf'),
                **metrics
            }

            return result

        except Exception as e:
            return {
                'config': config,
                'error': str(e),
                'ece': float('inf'),
                'nll': float('inf'),
                'accuracy': 0.0
            }

    def _fit_custom_optimizer(self, model, train_probs, train_labels, config):
        """Fit with custom optimizer (Adam or SGD)."""
        ablated_logits = torch.log(train_probs.clamp(1e-6, 1-1e-6))

        if config['optimizer'] == 'adam':
            optimizer = optim.Adam([model.temperature], lr=config['lr'])
        elif config['optimizer'] == 'sgd':
            optimizer = optim.SGD([model.temperature], lr=config['lr'])
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        criterion = nn.CrossEntropyLoss()

        stats = {
            "loss": [],
            "temperature": [],
        }

        for step in range(config['max_steps']):
            optimizer.zero_grad()
            scaled_logits = ablated_logits / model.temperature
            loss = criterion(scaled_logits, train_labels)
            loss.backward()
            optimizer.step()

            stats["loss"].append(loss.item())
            stats["temperature"].append(model.temperature.item())

            # Early stopping if loss becomes nan or temperature becomes too extreme
            if torch.isnan(loss) or model.temperature.item() < 0.001 or model.temperature.item() > 100:
                break

        model._is_fitted = True
        return stats

    def grid_search(self, train_probs, train_labels, val_probs, val_labels, param_grid):
        """Perform grid search over hyperparameters."""
        configs = list(ParameterGrid(param_grid))

        print(f"Running grid search with {len(configs)} configurations...")

        for config in tqdm(configs, desc="Grid search"):
            result = self.evaluate_config(
                train_probs, train_labels, val_probs, val_labels, config
            )
            self.results.append(result)

        return self.results

    def random_search(self, train_probs, train_labels, val_probs, val_labels,
                     param_distributions, n_iter=50):
        """Perform random search over hyperparameters."""
        print(f"Running random search with {n_iter} configurations...")

        for i in tqdm(range(n_iter), desc="Random search"):
            config = {}
            for param, distribution in param_distributions.items():
                if hasattr(distribution, 'rvs'):
                    config[param] = distribution.rvs()
                else:
                    config[param] = np.random.choice(distribution)

            result = self.evaluate_config(
                train_probs, train_labels, val_probs, val_labels, config
            )
            self.results.append(result)

        return self.results

    def get_best_config(self, metric='ece'):
        """Get the best configuration based on specified metric."""
        if not self.results:
            return None

        # Filter out failed results
        valid_results = [r for r in self.results if 'error' not in r]

        if not valid_results:
            print("No valid results found!")
            return None

        if metric == 'ece':
            best_result = min(valid_results, key=lambda x: x['ece'])
        elif metric == 'nll':
            best_result = min(valid_results, key=lambda x: x['nll'])
        elif metric == 'accuracy':
            best_result = max(valid_results, key=lambda x: x['accuracy'])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best_result


def load_sample_data(dataset='breast_cancer', n_samples=1000):
    """Load sample data for tuning."""
    if dataset == 'breast_cancer':
        # Import data setup function
        from experiments.tabular.breast_cancer_data_setup import load_breast_cancer_data

        # Load data with a single fraction for quick testing
        results = load_breast_cancer_data(
            model_type="vanilla",
            n_samples=n_samples,
            n_fractions=1,
            fill_value="mean"
        )

        # Extract train/val split from fraction 0
        train_probs = results['train_outputs'][0]  # Shape: (n_train, n_classes)
        train_labels = results['train_labels']

        # Use test as validation for hyperparameter tuning
        val_probs = results['test_outputs'][0]  # Shape: (n_test, n_classes)
        val_labels = results['test_labels']

        return train_probs, train_labels, val_probs, val_labels

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Temperature scaling hyperparameter tuning")
    parser.add_argument('--dataset', default='breast_cancer', choices=['breast_cancer', 'ctg', 'physionet'])
    parser.add_argument('--search_type', default='grid', choices=['grid', 'random', 'both'])
    parser.add_argument('--n_samples', type=int, default=500, help="Number of samples for quick tuning")
    parser.add_argument('--n_iter', type=int, default=50, help="Number of iterations for random search")
    parser.add_argument('--output_dir', default='./temperature_tuning_results')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_probs, train_labels, val_probs, val_labels = load_sample_data(
        dataset=args.dataset, n_samples=args.n_samples
    )

    # Convert to tensors
    train_probs = torch.tensor(train_probs, dtype=torch.float32, device=device)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)
    val_probs = torch.tensor(val_probs, dtype=torch.float32, device=device)
    val_labels = torch.tensor(val_labels, dtype=torch.long, device=device)

    num_classes = train_probs.shape[1]
    print(f"Data loaded: {train_probs.shape[0]} train, {val_probs.shape[0]} val, {num_classes} classes")

    # Initialize tuner
    tuner = TemperatureScalingTuner(num_classes=num_classes, device=device)

    # Define parameter spaces
    grid_params = {
        'lr': [0.001, 0.01, 0.1, 1.0],
        'max_steps': [100, 500, 1000, 2000],
        'optimizer': ['lbfgs', 'adam', 'sgd'],
        'temp_init': [0.5, 1.0, 2.0]
    }

    random_params = {
        'lr': loguniform(1e-4, 1.0),
        'max_steps': [100, 500, 1000, 2000, 5000],
        'optimizer': ['lbfgs', 'adam', 'sgd'],
        'temp_init': uniform(0.1, 5.0)
    }

    # Run searches
    if args.search_type in ['grid', 'both']:
        print("\n" + "="*50)
        print("GRID SEARCH")
        print("="*50)
        tuner.grid_search(train_probs, train_labels, val_probs, val_labels, grid_params)

    if args.search_type in ['random', 'both']:
        print("\n" + "="*50)
        print("RANDOM SEARCH")
        print("="*50)
        tuner.random_search(train_probs, train_labels, val_probs, val_labels,
                           random_params, n_iter=args.n_iter)

    # Analyze results
    print("\n" + "="*50)
    print("RESULTS ANALYSIS")
    print("="*50)

    for metric in ['ece', 'nll', 'accuracy']:
        best_result = tuner.get_best_config(metric=metric)
        if best_result:
            print(f"\nBest config for {metric.upper()}:")
            print(f"  Config: {best_result['config']}")
            print(f"  ECE: {best_result['ece']:.6f}")
            print(f"  NLL: {best_result['nll']:.6f}")
            print(f"  Accuracy: {best_result['accuracy']:.4f}")
            print(f"  Final Temperature: {best_result['final_temperature']:.4f}")

    # Save results
    output_file = os.path.join(args.output_dir, f"temperature_tuning_{args.dataset}_{args.search_type}.json")
    with open(output_file, 'w') as f:
        json.dump(tuner.results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    valid_results = [r for r in tuner.results if 'error' not in r]
    if valid_results:
        eces = [r['ece'] for r in valid_results]
        nlls = [r['nll'] for r in valid_results]
        accuracies = [r['accuracy'] for r in valid_results]

        print(f"\nSummary ({len(valid_results)} valid configurations):")
        print(f"  ECE: {np.mean(eces):.6f} ± {np.std(eces):.6f} (range: {np.min(eces):.6f} - {np.max(eces):.6f})")
        print(f"  NLL: {np.mean(nlls):.6f} ± {np.std(nlls):.6f} (range: {np.min(nlls):.6f} - {np.max(nlls):.6f})")
        print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f} (range: {np.min(accuracies):.4f} - {np.max(accuracies):.4f})")


if __name__ == "__main__":
    main()