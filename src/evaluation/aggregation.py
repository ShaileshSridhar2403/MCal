"""Result aggregation utilities for benchmarking."""

import numpy as np
from typing import Dict, List, Any, Optional
from tabulate import tabulate


def aggregate_fractionwise_kl(fractionwise_results: List[List[tuple]]) -> Dict[str, List[float]]:
    """Aggregate fractionwise KL divergence results across multiple runs.
    
    This function replicates the original aggregate_fractionwise_kl from get_benchmarks.py
    
    Args:
        fractionwise_results: List of lists of tuples (kl_argmax, kl_prob),
                             where each index corresponds to a fraction
    
    Returns:
        Dictionary containing mean and standard deviation for kl_argmax and kl_prob,
        organized by fraction
    """
    # Check if input is empty
    if not fractionwise_results or not fractionwise_results[0]:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": []}
    
    # Determine number of fractions
    num_fractions = len(fractionwise_results[0])
    
    # Initialize arrays to store values for each fraction across runs
    kl_argmax_values = [[] for _ in range(num_fractions)]
    kl_prob_values = [[] for _ in range(num_fractions)]
    
    # Collect values across all runs
    for run_results in fractionwise_results:
        for i, (kl_argmax, kl_prob) in enumerate(run_results):
            kl_argmax_values[i].append(kl_argmax)
            kl_prob_values[i].append(kl_prob)
    
    # Calculate mean and standard deviation for each fraction
    mean_argmax = [np.mean(values) for values in kl_argmax_values]
    std_argmax = [np.std(values) for values in kl_argmax_values]
    mean_prob = [np.mean(values) for values in kl_prob_values]
    std_prob = [np.std(values) for values in kl_prob_values]
    
    # Return results as a dictionary
    return {
        "mean_argmax": mean_argmax,
        "std_argmax": std_argmax,
        "mean_prob": mean_prob,
        "std_prob": std_prob
    }


def aggregate_results(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate results from multiple runs.
    
    This function replicates the original aggregate_results from get_benchmarks.py
    but is adapted for the MCal structure.
    
    Args:
        all_results: Dictionary of lists of result dictionaries, keyed by method
    
    Returns:
        Dictionary of aggregated results
    """
    aggregated_results = {}
    
    # Get all methods from the results
    methods = list(all_results.keys())
    
    for method in methods:
        # Skip if method doesn't have any results
        if not all_results[method]:
            continue
            
        num_runs = len(all_results[method])
        
        # Initialize method in aggregated results
        aggregated_results[method] = {}
        
        # Aggregate transformed KL results
        if 'kl_results_transformed' in all_results[method][0]:
            kl_transformed_results_prob = [
                all_results[method][i]['kl_results_transformed']['average_kl_prob'] 
                for i in range(num_runs) 
                if 'kl_results_transformed' in all_results[method][i]
            ]
            kl_transformed_results_onehot = [
                all_results[method][i]['kl_results_transformed']['average_kl_argmax'] 
                for i in range(num_runs)
                if 'kl_results_transformed' in all_results[method][i]
            ]
            
            if kl_transformed_results_prob and kl_transformed_results_onehot:
                aggregated_results[method].update({
                    'kl_transformed_mean_prob': np.mean(kl_transformed_results_prob),
                    'kl_transformed_std_prob': np.std(kl_transformed_results_prob),
                    'kl_transformed_mean_onehot': np.mean(kl_transformed_results_onehot),
                    'kl_transformed_std_onehot': np.std(kl_transformed_results_onehot),
                    'fraction_wise_results_transformed': aggregate_fractionwise_kl(
                        [all_results[method][i]['kl_values_transformed'] for i in range(num_runs)
                         if 'kl_values_transformed' in all_results[method][i]]
                    )
                })
        
        # Aggregate baseline KL results (for methods that provide baseline comparison)
        if 'kl_results_baseline' in all_results[method][0]:
            kl_baseline_results_prob = [
                all_results[method][i]['kl_results_baseline']['average_kl_prob'] 
                for i in range(num_runs)
                if 'kl_results_baseline' in all_results[method][i]
            ]
            kl_baseline_results_onehot = [
                all_results[method][i]['kl_results_baseline']['average_kl_argmax'] 
                for i in range(num_runs)
                if 'kl_results_baseline' in all_results[method][i]
            ]
            
            if kl_baseline_results_prob and kl_baseline_results_onehot:
                # Store baseline results separately to avoid duplication
                if 'baseline' not in aggregated_results:
                    aggregated_results['baseline'] = {
                        'kl_baseline_mean_prob': np.mean(kl_baseline_results_prob),
                        'kl_baseline_std_prob': np.std(kl_baseline_results_prob),
                        'kl_baseline_mean_onehot': np.mean(kl_baseline_results_onehot),
                        'kl_baseline_std_onehot': np.std(kl_baseline_results_onehot),
                        'fraction_wise_results': aggregate_fractionwise_kl(
                            [all_results[method][i]['kl_values_baseline'] for i in range(num_runs)
                             if 'kl_values_baseline' in all_results[method][i]]
                        )
                    }
        
        # Aggregate regular KL results (for methods without transformation)
        if ('kl_results' in all_results[method][0] and 
            'kl_results_transformed' not in all_results[method][0]):
            
            kl_results_prob = [
                all_results[method][i]['kl_results']['average_kl_prob'] 
                for i in range(num_runs)
                if 'kl_results' in all_results[method][i]
            ]
            kl_results_onehot = [
                all_results[method][i]['kl_results']['average_kl_argmax'] 
                for i in range(num_runs)
                if 'kl_results' in all_results[method][i]
            ]
            
            if kl_results_prob and kl_results_onehot:
                aggregated_results[method].update({
                    'kl_mean_prob': np.mean(kl_results_prob),
                    'kl_std_prob': np.std(kl_results_prob),
                    'kl_mean_onehot': np.mean(kl_results_onehot),
                    'kl_std_onehot': np.std(kl_results_onehot),
                    'fraction_wise_results': aggregate_fractionwise_kl(
                        [all_results[method][i]['kl_values'] for i in range(num_runs)
                         if 'kl_values' in all_results[method][i]]
                    )
                })
        
        # Aggregate training statistics if available
        training_stats_keys = []
        if all_results[method][0].get('training_stats'):
            training_stats_keys = all_results[method][0]['training_stats'].keys()
        
        if training_stats_keys:
            aggregated_training_stats = {}
            for stat_key in training_stats_keys:
                stat_values = []
                for i in range(num_runs):
                    if ('training_stats' in all_results[method][i] and 
                        stat_key in all_results[method][i]['training_stats']):
                        stat_value = all_results[method][i]['training_stats'][stat_key]
                        # Handle different types of statistics
                        if isinstance(stat_value, (int, float)):
                            stat_values.append(stat_value)
                        elif isinstance(stat_value, (list, np.ndarray)):
                            # For lists/arrays, take the final value
                            if len(stat_value) > 0:
                                stat_values.append(stat_value[-1])
                
                if stat_values:
                    aggregated_training_stats[stat_key] = {
                        'mean': np.mean(stat_values),
                        'std': np.std(stat_values),
                        'values': stat_values
                    }
            
            if aggregated_training_stats:
                aggregated_results[method]['training_stats'] = aggregated_training_stats
        
        # Aggregate calibration metrics if available
        if 'calibration_metrics' in all_results[method][0]:
            calibration_metrics = {}
            
            # Get keys from first run
            first_cal_metrics = all_results[method][0]['calibration_metrics']
            
            for metric_key in first_cal_metrics.keys():
                if metric_key == 'reliability_diagram':
                    # Skip reliability diagram for aggregation (too complex)
                    continue
                
                metric_values = []
                for i in range(num_runs):
                    if ('calibration_metrics' in all_results[method][i] and
                        metric_key in all_results[method][i]['calibration_metrics']):
                        value = all_results[method][i]['calibration_metrics'][metric_key]
                        if value is not None and not np.isnan(value):
                            metric_values.append(value)
                
                if metric_values:
                    calibration_metrics[metric_key] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values),
                        'values': metric_values
                    }
            
            if calibration_metrics:
                aggregated_results[method]['calibration_metrics'] = calibration_metrics
    
    return aggregated_results


def build_kl_comparison_table(
    aggregated_results: Dict[str, Any], 
    include_methods: Optional[List[str]] = None, 
    include_mean_baseline: bool = False
) -> str:
    """Build comparison table for KL divergence results with mean and standard deviation.
    
    This function replicates the original build_kl_comparison_table from get_benchmarks.py
    
    Args:
        aggregated_results: Dictionary of aggregated results
        include_methods: List of methods to include in the table
        include_mean_baseline: Whether to include baseline with fill_value="mean"
    
    Returns:
        Formatted table string
    """
    # Initialize table data
    table_data = [["Method", "Average KL (Prob)", "Average KL (Argmax)"]]
    
    # Define method display names
    method_names = {
        'baseline': "Original",
        'baseline_mean': "Replacement with dataset mean",
        'mcal': "MCal Calibration",
        'platt': "Platt Calibration", 
        'temperature': "Temperature Scaling",
        'optimized_lambda': "Optimized Lambda Transform",
        'expectation_prob': "Probability-based Transform",
        'expectation_onehot': "One-hot-based Transform",
        'patchcutout': "Training with PatchCutout",
        'patch_drop': "Patch Dropping",
        'neural': "Neural Transform",
        'logits_sharp': "Logits Sharp Transform",
        'logits_sharp_unconditioned': "Logits Sharp Unconditioned Transform"
    }
    
    # Add baseline if available
    if 'baseline' in aggregated_results:
        baseline = aggregated_results['baseline']
        table_data.append([
            method_names['baseline'], 
            f"{baseline['kl_baseline_mean_prob']:.2e} ± {baseline['kl_baseline_std_prob']:.2e}", 
            f"{baseline['kl_baseline_mean_onehot']:.2e} ± {baseline['kl_baseline_std_onehot']:.2e}"
        ])
    
    # Add mean baseline if available and requested
    if include_mean_baseline and 'baseline_mean' in aggregated_results:
        baseline_mean = aggregated_results['baseline_mean']
        if 'kl_baseline_mean_prob' in baseline_mean:
            table_data.append([
                method_names['baseline_mean'], 
                f"{baseline_mean['kl_baseline_mean_prob']:.2e} ± {baseline_mean['kl_baseline_std_prob']:.2e}", 
                f"{baseline_mean['kl_baseline_mean_onehot']:.2e} ± {baseline_mean['kl_baseline_std_onehot']:.2e}"
            ])
    
    # Methods to include in the table
    methods_to_include = include_methods or [
        m for m in aggregated_results.keys() 
        if m not in ['baseline', 'baseline_mean']
    ]
    
    # Add results for each method
    for method in methods_to_include:
        if method not in aggregated_results:
            continue
        
        method_data = aggregated_results[method]
        
        # Check for different result types and add accordingly
        if 'kl_mean_prob' in method_data:
            # Methods without transformation (like PatchCutout)
            table_data.append([
                method_names.get(method, method.replace('_', ' ').title()), 
                f"{method_data['kl_mean_prob']:.2e} ± {method_data['kl_std_prob']:.2e}", 
                f"{method_data['kl_mean_onehot']:.2e} ± {method_data['kl_std_onehot']:.2e}"
            ])
        elif 'kl_transformed_mean_prob' in method_data:
            # Methods with transformation
            table_data.append([
                method_names.get(method, method.replace('_', ' ').title()), 
                f"{method_data['kl_transformed_mean_prob']:.2e} ± {method_data['kl_transformed_std_prob']:.2e}", 
                f"{method_data['kl_transformed_mean_onehot']:.2e} ± {method_data['kl_transformed_std_onehot']:.2e}"
            ])
    
    # Generate table
    table = tabulate(table_data, headers="firstrow", tablefmt="grid")
    return table


def build_calibration_comparison_table(aggregated_results: Dict[str, Any]) -> str:
    """Build comparison table for calibration metrics.
    
    Args:
        aggregated_results: Dictionary of aggregated results
        
    Returns:
        Formatted table string
    """
    # Initialize table data
    table_data = [["Method", "ECE", "MCE", "Brier Score", "Log Loss"]]
    
    # Define method display names
    method_names = {
        'mcal': "MCal Calibration",
        'platt': "Platt Calibration", 
        'temperature': "Temperature Scaling",
        'optimized_lambda': "Optimized Lambda Transform",
        'expectation_prob': "Probability-based Transform",
        'expectation_onehot': "One-hot-based Transform",
        'neural': "Neural Transform",
        'logits_sharp': "Logits Sharp Transform"
    }
    
    for method, results in aggregated_results.items():
        if method in ['baseline', 'baseline_mean']:
            continue
            
        if 'calibration_metrics' not in results:
            continue
        
        cal_metrics = results['calibration_metrics']
        
        # Extract metrics with error handling
        ece = cal_metrics.get('ece', {})
        mce = cal_metrics.get('mce', {})
        brier = cal_metrics.get('brier_score', {})
        log_loss = cal_metrics.get('log_loss', {})
        
        ece_str = f"{ece.get('mean', 0):.3f} ± {ece.get('std', 0):.3f}" if ece else "N/A"
        mce_str = f"{mce.get('mean', 0):.3f} ± {mce.get('std', 0):.3f}" if mce else "N/A"
        brier_str = f"{brier.get('mean', 0):.3f} ± {brier.get('std', 0):.3f}" if brier else "N/A"
        log_loss_str = f"{log_loss.get('mean', 0):.3f} ± {log_loss.get('std', 0):.3f}" if log_loss else "N/A"
        
        table_data.append([
            method_names.get(method, method.replace('_', ' ').title()),
            ece_str,
            mce_str, 
            brier_str,
            log_loss_str
        ])
    
    # Generate table
    table = tabulate(table_data, headers="firstrow", tablefmt="grid")
    return table


def build_cross_dataset_summary_table(
    all_results: Dict[str, Dict[str, Any]], 
    methods: List[str]
) -> str:
    """Build a summary table across multiple datasets.
    
    Args:
        all_results: Results for all datasets
        methods: Methods to include
        
    Returns:
        Formatted summary table string
    """
    # Initialize table data
    table_data = [["Method"] + list(all_results.keys()) + ["Average"]]
    
    # Define method display names
    method_names = {
        'baseline': "Original",
        'mcal': "MCal Calibration",
        'platt': "Platt Calibration", 
        'temperature': "Temperature Scaling",
        'optimized_lambda': "Optimized Lambda Transform",
        'expectation_prob': "Probability-based Transform",
        'expectation_onehot': "One-hot-based Transform",
        'neural': "Neural Transform",
        'logits_sharp': "Logits Sharp Transform"
    }
    
    # Add baseline first if available
    if all(('baseline' in results for results in all_results.values())):
        baseline_values = []
        for dataset_name, results in all_results.items():
            baseline = results['baseline']
            kl_prob = baseline['kl_baseline_mean_prob']
            baseline_values.append(kl_prob)
        
        avg_baseline = np.mean(baseline_values)
        row = [method_names['baseline']] + [f"{val:.2e}" for val in baseline_values] + [f"{avg_baseline:.2e}"]
        table_data.append(row)
    
    # Add each method
    for method in methods:
        if method in ['baseline', 'baseline_mean']:
            continue
            
        method_values = []
        for dataset_name, results in all_results.items():
            if method not in results:
                method_values.append(float('inf'))
                continue
                
            method_data = results[method]
            
            # Get the appropriate KL value
            if 'kl_transformed_mean_prob' in method_data:
                kl_value = method_data['kl_transformed_mean_prob']
            elif 'kl_mean_prob' in method_data:
                kl_value = method_data['kl_mean_prob']
            else:
                kl_value = float('inf')
            
            method_values.append(kl_value)
        
        # Calculate average (excluding inf values)
        finite_values = [v for v in method_values if np.isfinite(v)]
        avg_value = np.mean(finite_values) if finite_values else float('inf')
        
        # Format row
        formatted_values = []
        for val in method_values:
            if np.isfinite(val):
                formatted_values.append(f"{val:.2e}")
            else:
                formatted_values.append("Failed")
        
        avg_str = f"{avg_value:.2e}" if np.isfinite(avg_value) else "N/A"
        row = [method_names.get(method, method.replace('_', ' ').title())] + formatted_values + [avg_str]
        table_data.append(row)
    
    # Generate table
    table = tabulate(table_data, headers="firstrow", tablefmt="grid")
    return table


def compute_method_rankings(
    aggregated_results: Dict[str, Any],
    metric: str = 'kl_transformed_mean_prob'
) -> List[tuple]:
    """Compute rankings of methods based on a specific metric.
    
    Args:
        aggregated_results: Dictionary of aggregated results
        metric: Metric to rank by (lower is better)
        
    Returns:
        List of (method_name, metric_value) tuples sorted by performance
    """
    rankings = []
    
    for method, results in aggregated_results.items():
        if method in ['baseline', 'baseline_mean']:
            continue
            
        # Get the metric value
        if metric in results:
            value = results[metric]
        elif metric.replace('_transformed', '') in results:
            # Fallback to non-transformed version
            value = results[metric.replace('_transformed', '')]
        else:
            continue
        
        rankings.append((method, value))
    
    # Sort by metric value (lower is better for KL divergence)
    rankings.sort(key=lambda x: x[1])
    
    return rankings


def compute_statistical_significance(
    results1: List[float],
    results2: List[float],
    test: str = 'ttest'
) -> Dict[str, Any]:
    """Compute statistical significance between two sets of results.
    
    Args:
        results1: First set of results
        results2: Second set of results  
        test: Statistical test to use ('ttest', 'wilcoxon', 'mannwhitney')
        
    Returns:
        Dictionary containing test results
    """
    from scipy import stats
    
    results1 = np.array(results1)
    results2 = np.array(results2)
    
    # Remove any NaN or infinite values
    mask1 = np.isfinite(results1)
    mask2 = np.isfinite(results2)
    results1_clean = results1[mask1]
    results2_clean = results2[mask2]
    
    if len(results1_clean) == 0 or len(results2_clean) == 0:
        return {'test': test, 'statistic': None, 'p_value': None, 'significant': False}
    
    if test == 'ttest':
        statistic, p_value = stats.ttest_ind(results1_clean, results2_clean)
    elif test == 'wilcoxon':
        if len(results1_clean) == len(results2_clean):
            statistic, p_value = stats.wilcoxon(results1_clean, results2_clean)
        else:
            # Fall back to Mann-Whitney if lengths don't match
            statistic, p_value = stats.mannwhitneyu(results1_clean, results2_clean)
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(results1_clean, results2_clean)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Determine significance (p < 0.05)
    significant = p_value < 0.05 if p_value is not None else False
    
    return {
        'test': test,
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'n1': len(results1_clean),
        'n2': len(results2_clean)
    }