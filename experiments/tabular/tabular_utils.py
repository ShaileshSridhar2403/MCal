#!/usr/bin/env python3
"""
Common Tabular Utilities for MCal Benchmarks

This module provides common utilities for tabular data processing,
missing data simulation, and result aggregation.
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt


def randomly_remove_data(X, fraction, method="feature_wise"):
    """
    Randomly remove data from features or samples.

    Args:
        X (pd.DataFrame or np.ndarray): Input data
        fraction (float): Fraction of data to remove (0 to 1)
        method (str): "feature_wise" or "sample_wise"

    Returns:
        Data with randomly removed values
    """
    if isinstance(X, pd.DataFrame):
        X_removed = X.copy()

        if method == "feature_wise":
            # Remove fraction of values from each feature
            for column in X_removed.columns:
                mask = np.random.random(len(X_removed)) < fraction
                X_removed.loc[mask, column] = np.nan
        elif method == "sample_wise":
            # Remove fraction of features from each sample
            n_features = len(X_removed.columns)
            for idx in range(len(X_removed)):
                n_remove = int(n_features * fraction)
                if n_remove > 0:
                    features_to_remove = np.random.choice(
                        X_removed.columns, size=n_remove, replace=False
                    )
                    X_removed.loc[idx, features_to_remove] = np.nan
    else:
        X_removed = X.copy()
        n_samples, n_features = X.shape

        if method == "feature_wise":
            # Remove fraction of values from each feature
            for j in range(n_features):
                mask = np.random.random(n_samples) < fraction
                X_removed[mask, j] = np.nan
        elif method == "sample_wise":
            # Remove fraction of features from each sample
            for i in range(n_samples):
                n_remove = int(n_features * fraction)
                if n_remove > 0:
                    features_to_remove = np.random.choice(
                        n_features, size=n_remove, replace=False
                    )
                    X_removed[i, features_to_remove] = np.nan

    return X_removed


def calculate_missingness_statistics(X):
    """
    Calculate missingness statistics for a dataset.

    Args:
        X (pd.DataFrame): Dataset

    Returns:
        dict: Missingness statistics
    """
    stats = {}

    # Overall missingness
    total_cells = X.size
    missing_cells = X.isnull().sum().sum()
    stats['overall_missingness'] = missing_cells / total_cells

    # Feature-wise missingness
    feature_missingness = X.isnull().sum() / len(X)
    stats['feature_missingness'] = {
        'mean': feature_missingness.mean(),
        'std': feature_missingness.std(),
        'min': feature_missingness.min(),
        'max': feature_missingness.max()
    }

    # Sample-wise missingness
    sample_missingness = X.isnull().sum(axis=1) / X.shape[1]
    stats['sample_missingness'] = {
        'mean': sample_missingness.mean(),
        'std': sample_missingness.std(),
        'min': sample_missingness.min(),
        'max': sample_missingness.max()
    }

    return stats


def aggregate_fractionwise_kl(fractionwise_results):
    """
    Aggregate fractionwise KL divergence and accuracy results across multiple runs.

    Args:
        fractionwise_results: List of run results containing kl_values_argmax, kl_values_prob, and kl_values_accuracy

    Returns:
        dict: Aggregated fractionwise results
    """
    if not fractionwise_results or not fractionwise_results[0]:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": [], "mean_accuracy": [], "std_accuracy": []}

    # Determine number of fractions
    first_result = fractionwise_results[0]
    if isinstance(first_result, dict) and 'kl_values_argmax' in first_result:
        num_fractions = len(first_result['kl_values_argmax'])
    else:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": [], "mean_accuracy": [], "std_accuracy": []}

    # Initialize arrays for each fraction
    kl_argmax_values = [[] for _ in range(num_fractions)]
    kl_prob_values = [[] for _ in range(num_fractions)]
    accuracy_values = [[] for _ in range(num_fractions)]

    # Collect values across all runs
    for run_results in fractionwise_results:
        kl_argmax_list = run_results['kl_values_argmax']
        kl_prob_list = run_results['kl_values_prob']
        accuracy_list = run_results.get('kl_values_accuracy', [])

        for i in range(min(len(kl_argmax_list), num_fractions)):
            kl_argmax_values[i].append(kl_argmax_list[i])
            kl_prob_values[i].append(kl_prob_list[i])
            if i < len(accuracy_list):
                accuracy_values[i].append(accuracy_list[i])

    # Calculate mean and standard deviation
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
    """
    Aggregate results across multiple runs following MCal pattern.

    Args:
        all_results (dict): Dictionary containing results for each method

    Returns:
        dict: Aggregated results
    """
    aggregated_results = {}

    for method, results in all_results.items():
        if not results:
            continue

        # Extract values across runs
        kl_prob_values = [r['average_kl_prob'] for r in results]
        kl_argmax_values = [r['average_kl_argmax'] for r in results]
        accuracy_values = [r.get('average_accuracy', 0) for r in results if r and 'average_accuracy' in r]

        # Aggregate fraction-wise results
        fraction_wise_results = aggregate_fractionwise_kl(results)

        aggregated_results[method] = {
            'kl_transformed_mean_prob': np.mean(kl_prob_values),
            'kl_transformed_std_prob': np.std(kl_prob_values),
            'kl_transformed_mean_onehot': np.mean(kl_argmax_values),
            'kl_transformed_std_onehot': np.std(kl_argmax_values),
            'accuracy_transformed_mean': np.mean(accuracy_values) if accuracy_values else 0.0,
            'accuracy_transformed_std': np.std(accuracy_values) if len(accuracy_values) > 1 else 0.0,
            'fraction_wise_results_transformed': fraction_wise_results
        }

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


def build_kl_comparison_table(aggregated_results, include_methods=None, dataset_name="Tabular"):
    """
    Build comparison table for KL divergence results.

    Args:
        aggregated_results (dict): Aggregated results
        include_methods (list): Methods to include in table
        dataset_name (str): Dataset name for display

    Returns:
        str: Formatted comparison table
    """
    # Initialize table data
    table_data = [["Method", "Average KL (Prob)", "Average KL (Argmax)"]]

    # Define method display names
    method_names = {
        'baseline': "Original",
        'mcal': "MCal (Vector Scaling)",
        'mcal_ce': "MCal_CE (Cross-Entropy)",
        'mcal_ce_uncond': "MCal_CE_Uncond (Unconditional)",
        'platt': "Platt Scaling",
        'temperature': "Temperature Scaling",
        'logits_sharp': "LogitsSharp Transform",
        'mean_imputation': "Mean Imputation Model",
        'zero_imputation': "Zero Imputation Model",
        'xgboost_native': "XGBoost Native Missing"
    }

    # Add baseline if available
    if 'baseline' in aggregated_results:
        baseline = aggregated_results['baseline']
        table_data.append([
            method_names['baseline'],
            f"{baseline['kl_baseline_mean_prob']:.2e} ± {baseline['kl_baseline_std_prob']:.2e}",
            f"{baseline['kl_baseline_mean_onehot']:.2e} ± {baseline['kl_baseline_std_onehot']:.2e}"
        ])

    # Methods to include in the table
    methods_to_include = include_methods or [m for m in aggregated_results.keys() if m != 'baseline']

    # Add results for each method
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

    # Generate table
    table = tabulate(table_data, headers="firstrow", tablefmt="grid")
    return table


def plot_kl_divergence(aggregated_results, dataset_name, save_path=None):
    """
    Plot KL divergence results across fractions.

    Args:
        aggregated_results (dict): Aggregated results
        dataset_name (str): Dataset name
        save_path (str): Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    markers = ['o', 's', '^', 'D', 'v', '<']

    # Get fractions (assuming consistent across methods)
    fractions = None
    methods_with_data = []

    for method, result in aggregated_results.items():
        if 'fraction_wise_results_transformed' in result:
            if fractions is None:
                fractions = np.linspace(0, 0.9, len(result['fraction_wise_results_transformed']['mean_prob']))
            methods_with_data.append(method)

    if fractions is None:
        print("No fractionwise data found for plotting")
        return

    # Plot each method
    for i, method in enumerate(methods_with_data[:len(colors)]):
        result = aggregated_results[method]
        fr_results = result['fraction_wise_results_transformed']

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Plot probability KL divergence
        mean_prob = fr_results['mean_prob']
        std_prob = fr_results['std_prob']
        ax1.errorbar(fractions, mean_prob, yerr=std_prob,
                    label=method.replace('_', ' ').title(),
                    color=color, marker=marker, capsize=5)

        # Plot argmax KL divergence
        mean_argmax = fr_results['mean_argmax']
        std_argmax = fr_results['std_argmax']
        ax2.errorbar(fractions, mean_argmax, yerr=std_argmax,
                    label=method.replace('_', ' ').title(),
                    color=color, marker=marker, capsize=5)

    # Configure plots
    ax1.set_title(f'KL Divergence (Probability) - {dataset_name}')
    ax1.set_xlabel('Missing Data Fraction')
    ax1.set_ylabel('KL Divergence (Probability)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.set_title(f'KL Divergence (Argmax) - {dataset_name}')
    ax2.set_xlabel('Missing Data Fraction')
    ax2.set_ylabel('KL Divergence (Argmax)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


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


def save_results(aggregated_results, save_dir, dataset_name, n_runs):
    """
    Save aggregated results to JSON and generate comparison table.

    Args:
        aggregated_results (dict): Aggregated results
        save_dir (str): Directory to save results
        dataset_name (str): Dataset name
        n_runs (int): Number of runs
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)

    # Save results as JSON
    json_path = os.path.join(save_dir, "json", f"aggregated_results_{dataset_name.lower()}.json")
    json_serializable_results = convert_to_json_serializable(aggregated_results)

    with open(json_path, 'w') as f:
        json.dump(json_serializable_results, f, indent=4)
    print(f"Aggregated results saved to {json_path}")

    # Build and display comparison table
    table = build_kl_comparison_table(aggregated_results, dataset_name=dataset_name)

    print(f"\nKL Divergence Comparison for {dataset_name} (averaged over {n_runs} runs):")
    print(table)

    # Save table
    table_path = os.path.join(save_dir, f"kl_comparison_table_{dataset_name.lower()}.txt")
    with open(table_path, 'w') as f:
        f.write(f"KL Divergence Comparison for {dataset_name} (averaged over {n_runs} runs):\n")
        f.write(table)
    print(f"Comparison table saved to {table_path}")

    # Generate and save plot
    plot_path = os.path.join(save_dir, f"kl_divergence_{dataset_name.lower()}.png")
    plot_kl_divergence(aggregated_results, dataset_name, save_path=plot_path)

    return json_path, table_path, plot_path


def validate_dataset_structure(X, y, dataset_name="Dataset"):
    """
    Validate dataset structure and provide summary statistics.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        dataset_name (str): Dataset name for logging

    Returns:
        dict: Validation summary
    """
    print(f"\n=== {dataset_name} Dataset Validation ===")

    summary = {
        'n_samples': len(X),
        'n_features': len(X.columns) if hasattr(X, 'columns') else X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': {},
        'missing_data': {}
    }

    # Basic statistics
    print(f"Samples: {summary['n_samples']}")
    print(f"Features: {summary['n_features']}")
    print(f"Classes: {summary['n_classes']}")

    # Class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        summary['class_distribution'][str(cls)] = int(count)
        print(f"  Class {cls}: {count} ({count/len(y)*100:.1f}%)")

    # Missing data analysis
    if isinstance(X, pd.DataFrame):
        missing_stats = calculate_missingness_statistics(X)
        summary['missing_data'] = missing_stats
        print(f"Missing data: {missing_stats['overall_missingness']*100:.1f}%")

    print("=" * (len(dataset_name) + 25))

    return summary


def create_dataset_report(X, y, dataset_name, save_path=None):
    """
    Create comprehensive dataset report.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        dataset_name (str): Dataset name
        save_path (str): Path to save report

    Returns:
        str: Report content
    """
    # Validate dataset
    summary = validate_dataset_structure(X, y, dataset_name)

    # Generate report
    report = f"""
# {dataset_name} Dataset Report

## Dataset Overview
- **Samples**: {summary['n_samples']:,}
- **Features**: {summary['n_features']:,}
- **Classes**: {summary['n_classes']}
- **Task Type**: {'Binary Classification' if summary['n_classes'] == 2 else 'Multi-class Classification'}

## Class Distribution
"""

    for cls, count in summary['class_distribution'].items():
        percentage = count / summary['n_samples'] * 100
        report += f"- **Class {cls}**: {count:,} samples ({percentage:.1f}%)\n"

    if 'missing_data' in summary and summary['missing_data']:
        missing = summary['missing_data']
        report += f"""
## Missing Data Analysis
- **Overall Missingness**: {missing['overall_missingness']*100:.2f}%
- **Feature-wise Missingness**:
  - Mean: {missing['feature_missingness']['mean']*100:.2f}%
  - Std: {missing['feature_missingness']['std']*100:.2f}%
  - Range: {missing['feature_missingness']['min']*100:.2f}% - {missing['feature_missingness']['max']*100:.2f}%
- **Sample-wise Missingness**:
  - Mean: {missing['sample_missingness']['mean']*100:.2f}%
  - Std: {missing['sample_missingness']['std']*100:.2f}%
  - Range: {missing['sample_missingness']['min']*100:.2f}% - {missing['sample_missingness']['max']*100:.2f}%
"""

    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Dataset report saved to {save_path}")

    return report


if __name__ == "__main__":
    # Test tabular utilities
    print("Testing tabular utilities...")

    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randint(0, 2, n_samples))

    try:
        # Test missing data simulation
        X_missing = randomly_remove_data(X, 0.3)
        print(f"✓ Missing data simulation: {X_missing.isnull().sum().sum()} missing values")

        # Test missingness statistics
        stats = calculate_missingness_statistics(X_missing)
        print(f"✓ Missingness statistics: {stats['overall_missingness']:.2f} overall")

        # Test dataset validation
        summary = validate_dataset_structure(X, y, "Test Dataset")
        print("✓ Dataset validation successful")

        # Test dataset report
        report = create_dataset_report(X, y, "Test Dataset")
        print("✓ Dataset report generated")

        print("✓ All tabular utility tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()