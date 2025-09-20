#!/usr/bin/env python3
"""
Demo script for MCal Tabular Benchmarks

This script demonstrates how to use the tabular benchmark with synthetic data
when PhysioNet data is not available.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from xgboost_utils import MCALXGBoostPredictor
from tabular_kl_benchmark import calculate_kl_metrics, apply_transform
# Import from our local tabular_utils (not XAI_Benchmark's)
sys.path.insert(0, str(current_dir))  # Ensure local imports take precedence
from tabular_utils import (
    randomly_remove_data,
    aggregate_results,
    build_kl_comparison_table,
    save_results
)


def create_synthetic_medical_data(n_samples=1000, n_features=20, missing_fraction=0.1):
    """
    Create synthetic medical-like tabular data for demonstration.

    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        missing_fraction (float): Fraction of missing data to introduce

    Returns:
        tuple: (X, y) where X is features and y is binary labels
    """
    np.random.seed(42)  # For reproducibility

    # Generate features with different distributions to mimic medical data
    X = pd.DataFrame()

    # Normal vital signs (e.g., heart rate, blood pressure)
    X['heart_rate'] = np.random.normal(70, 15, n_samples)
    X['systolic_bp'] = np.random.normal(120, 20, n_samples)
    X['diastolic_bp'] = np.random.normal(80, 10, n_samples)
    X['temperature'] = np.random.normal(98.6, 1.5, n_samples)
    X['oxygen_sat'] = np.random.normal(98, 2, n_samples)

    # Lab values (often log-normal)
    X['white_blood_cells'] = np.random.lognormal(2, 0.5, n_samples)
    X['red_blood_cells'] = np.random.lognormal(1.5, 0.3, n_samples)
    X['glucose'] = np.random.lognormal(4.6, 0.3, n_samples)
    X['creatinine'] = np.random.lognormal(0, 0.4, n_samples)

    # Binary indicators (e.g., medications, conditions)
    for i in range(10, n_features):
        X[f'binary_feature_{i}'] = np.random.binomial(1, 0.3, n_samples)

    # Create labels with some correlation to features
    # Higher risk if abnormal vital signs or lab values
    risk_score = (
        (X['heart_rate'] > 100).astype(int) +
        (X['systolic_bp'] > 140).astype(int) +
        (X['temperature'] > 100).astype(int) +
        (X['white_blood_cells'] > 12).astype(int) +
        (X['glucose'] > 140).astype(int)
    )

    # Convert risk score to binary outcome with some noise
    y = pd.Series((risk_score >= 2).astype(int))

    # Add some random noise to labels
    noise_mask = np.random.random(n_samples) < 0.1  # 10% label noise
    y[noise_mask] = 1 - y[noise_mask]

    # Introduce missing data
    if missing_fraction > 0:
        X = randomly_remove_data(X, missing_fraction)

    print(f"Created synthetic medical dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Missing data: {X.isnull().sum().sum() / X.size * 100:.1f}%")
    print(f"  Class distribution: {y.value_counts().to_dict()}")

    return X, y


def demo_xgboost_training():
    """Demonstrate XGBoost training with missing data."""
    print("\n" + "="*60)
    print("DEMO: XGBoost Training with Missing Data")
    print("="*60)

    # Create synthetic data
    X, y = create_synthetic_medical_data(n_samples=500, missing_fraction=0.2)

    # Train XGBoost model with missingness robustness
    print("\nTraining XGBoost model...")
    predictor = MCALXGBoostPredictor(imputation_strategy="mean")
    predictor.fit(X, y, n_epochs=3)  # Minimal training for demo

    # Generate fractionwise predictions
    print("\nGenerating fractionwise predictions...")
    test_X = X.iloc[:100].copy()  # Use subset for demo
    removal_fractions = [0.0, 0.3, 0.6, 0.9]

    predictions = predictor.predict_fractionwise(test_X, removal_fractions)
    print(f"Predictions shape: {predictions.shape}")

    return predictions, y.iloc[:100].values


def demo_kl_calculation():
    """Demonstrate KL divergence calculation."""
    print("\n" + "="*60)
    print("DEMO: KL Divergence Calculation")
    print("="*60)

    # Generate predictions
    predictions, labels = demo_xgboost_training()

    # Calculate KL metrics
    print("\nCalculating KL divergence metrics...")
    kl_results = calculate_kl_metrics(predictions, labels)

    print(f"\nResults:")
    print(f"  Average KL (Probability): {kl_results['average_kl_prob']:.6f}")
    print(f"  Average KL (Argmax): {kl_results['average_kl_argmax']:.6f}")
    print(f"  Average Accuracy: {kl_results['average_accuracy']:.4f}")

    return predictions, labels


def demo_calibration_methods():
    """Demonstrate calibration methods."""
    print("\n" + "="*60)
    print("DEMO: Calibration Methods")
    print("="*60)

    # Generate predictions
    predictions, labels = demo_xgboost_training()
    device = torch.device("cpu")

    # Test different calibration methods
    methods = ['baseline', 'mcal', 'platt', 'temperature']
    results = {}

    for method in methods:
        print(f"\nTesting {method} calibration...")

        try:
            # Apply calibration
            method_kwargs = {'max_steps': 100}  # Fast for demo
            if method == 'mcal':
                method_kwargs['kappa'] = 2.0

            calibrated_predictions = apply_transform(
                predictions, labels, method, device, **method_kwargs
            )

            # Calculate KL metrics
            kl_results = calculate_kl_metrics(calibrated_predictions, labels)
            results[method] = [kl_results]  # List format for aggregation

            print(f"  KL (Prob): {kl_results['average_kl_prob']:.6f}")
            print(f"  KL (Argmax): {kl_results['average_kl_argmax']:.6f}")
            print(f"  Accuracy: {kl_results['average_accuracy']:.4f}")

        except Exception as e:
            print(f"  Error: {str(e)}")
            results[method] = []

    return results


def demo_results_aggregation():
    """Demonstrate results aggregation and table generation."""
    print("\n" + "="*60)
    print("DEMO: Results Aggregation")
    print("="*60)

    # Get calibration results
    all_results = demo_calibration_methods()

    # Aggregate results (simulate multiple runs by duplicating)
    print("\nAggregating results...")
    for method in all_results:
        if all_results[method]:
            # Simulate 3 runs by adding small noise
            base_result = all_results[method][0]
            for i in range(2):  # Add 2 more "runs"
                noisy_result = base_result.copy()
                noise_factor = 0.1 * np.random.randn()
                noisy_result['average_kl_prob'] *= (1 + noise_factor)
                noisy_result['average_kl_argmax'] *= (1 + noise_factor)
                all_results[method].append(noisy_result)

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # Build comparison table
    table = build_kl_comparison_table(aggregated, dataset_name="Synthetic Medical")
    print("\nComparison Table:")
    print(table)

    return aggregated


def main():
    """Run the complete demo."""
    print("="*60)
    print("MCal Tabular Benchmark Demo")
    print("="*60)
    print("\nThis demo shows how to use the tabular benchmark")
    print("with synthetic medical data when PhysioNet is not available.")

    try:
        # Run all demo sections
        demo_xgboost_training()
        demo_kl_calculation()
        demo_calibration_methods()
        aggregated_results = demo_results_aggregation()

        # Save demo results
        print("\n" + "="*60)
        print("DEMO: Saving Results")
        print("="*60)

        save_dir = "/tmp/mcal_tabular_demo"
        os.makedirs(save_dir, exist_ok=True)

        json_path, table_path, plot_path = save_results(
            aggregated_results, save_dir, "SyntheticMedical", n_runs=3
        )

        print(f"\nDemo completed successfully!")
        print(f"Results saved to: {save_dir}")
        print(f"- JSON: {json_path}")
        print(f"- Table: {table_path}")
        print(f"- Plot: {plot_path}")

    except Exception as e:
        print(f"\nDemo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 Tabular benchmark demo completed successfully!")
    print("\nTo run with real PhysioNet data:")
    print("  python tabular_kl_benchmark.py --samples 100 --runs 1")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)