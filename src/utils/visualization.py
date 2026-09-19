"""Visualization utilities for calibration analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple
import torch
from pathlib import Path


def plot_kl_divergence(
    aggregated_results: dict,
    dataset: str,
    save_path: Optional[str] = None,
    include_methods: Optional[List[str]] = None,
    include_mean_baseline: bool = False
) -> None:
    """Plot KL divergence values with error bars for different methods.
    
    This function replicates the original plot_kl_divergence from get_benchmarks.py
    Creates three plots: a combined plot with both metrics and two separate plots.
    
    Args:
        aggregated_results: Dictionary of aggregated results
        dataset: Dataset name for plot title
        save_path: Path to save the plot (without extension)
        include_methods: List of methods to include in the plot
        include_mean_baseline: Whether to include baseline with fill_value="mean"
    """
    # Define colors and markers
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', '*', 'p', 'x', '+', 'v', '<']
    
    # Define fractions (assuming 16 fractions from 0 to 1)
    fractions = np.linspace(0, 1, 16)
    
    # Methods to include in the plot
    methods_to_include = include_methods or [m for m in aggregated_results.keys() 
                                           if m not in ['baseline', 'baseline_mean']]
    
    # Create the list of methods to plot
    methods_to_plot = []
    if 'baseline' in aggregated_results:
        methods_to_plot.append('baseline')
    if include_mean_baseline and 'baseline_mean' in aggregated_results:
        methods_to_plot.append('baseline_mean')
    methods_to_plot.extend(methods_to_include)
    
    # Method display names mapping
    method_display_names = {
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
        'logits_sharp': "Logits Sharp Transform",
        'logits_sharp_unconditioned': "Logits Sharp Unconditioned Transform"
    }
    
    # 1. Create the combined plot (original behavior)
    fig_combined, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # 2. Create separate plots
    fig_prob = plt.figure(figsize=(12, 8))
    ax_prob = fig_prob.add_subplot(111)
    
    fig_argmax = plt.figure(figsize=(12, 8))
    ax_argmax = fig_argmax.add_subplot(111)
    
    # Plot data for probability KL divergence (on both combined and separate plots)
    for i, (method, color, marker) in enumerate(zip(
        methods_to_plot, 
        colors[:len(methods_to_plot)], 
        markers[:len(methods_to_plot)])):
        
        if method not in aggregated_results:
            continue
            
        # Get the appropriate results key
        if method == 'baseline' and 'fraction_wise_results' in aggregated_results[method]:
            fr_results = aggregated_results[method]['fraction_wise_results']
        elif method == 'baseline_mean' and 'fraction_wise_results' in aggregated_results[method]:
            fr_results = aggregated_results[method]['fraction_wise_results']
        elif method in ['patchcutout', 'patch_drop'] and 'fraction_wise_results' in aggregated_results[method]:
            fr_results = aggregated_results[method]['fraction_wise_results']
        elif 'fraction_wise_results_transformed' in aggregated_results[method]:
            fr_results = aggregated_results[method]['fraction_wise_results_transformed']
        else:
            continue
        
        mean_prob = fr_results['mean_prob']
        std_prob = fr_results['std_prob']
        
        # Get display name for the method
        display_name = method_display_names.get(method, method.replace('_', ' ').title())
        
        # Ensure we have the right number of fractions
        if len(mean_prob) != len(fractions):
            # Adjust fractions to match data
            fractions_adjusted = np.linspace(0, 1, len(mean_prob))
        else:
            fractions_adjusted = fractions
        
        # Plot on combined plot
        ax1.errorbar(fractions_adjusted, mean_prob, yerr=std_prob, 
                    label=display_name, 
                    color=color, marker=marker, capsize=5, markersize=8, 
                    linewidth=2, elinewidth=1)
        
        # Plot on separate probability plot
        ax_prob.errorbar(fractions_adjusted, mean_prob, yerr=std_prob, 
                        label=display_name, 
                        color=color, marker=marker, capsize=5, markersize=8, 
                        linewidth=2, elinewidth=1)
    
    # Plot data for argmax KL divergence (on both combined and separate plots)
    for i, (method, color, marker) in enumerate(zip(
        methods_to_plot, 
        colors[:len(methods_to_plot)], 
        markers[:len(methods_to_plot)])):
        
        if method not in aggregated_results:
            continue
            
        # Get the appropriate results key
        if method == 'baseline' and 'fraction_wise_results' in aggregated_results[method]:
            fr_results = aggregated_results[method]['fraction_wise_results']
        elif method == 'baseline_mean' and 'fraction_wise_results' in aggregated_results[method]:
            fr_results = aggregated_results[method]['fraction_wise_results']
        elif method in ['patchcutout', 'patch_drop'] and 'fraction_wise_results' in aggregated_results[method]:
            fr_results = aggregated_results[method]['fraction_wise_results']
        elif 'fraction_wise_results_transformed' in aggregated_results[method]:
            fr_results = aggregated_results[method]['fraction_wise_results_transformed']
        else:
            continue
        
        mean_argmax = fr_results['mean_argmax']
        std_argmax = fr_results['std_argmax']
        
        # Get display name for the method
        display_name = method_display_names.get(method, method.replace('_', ' ').title())
        
        # Ensure we have the right number of fractions
        if len(mean_argmax) != len(fractions):
            # Adjust fractions to match data
            fractions_adjusted = np.linspace(0, 1, len(mean_argmax))
        else:
            fractions_adjusted = fractions
        
        # Plot on combined plot
        ax2.errorbar(fractions_adjusted, mean_argmax, yerr=std_argmax, 
                    label=display_name, 
                    color=color, marker=marker, capsize=5, markersize=8, 
                    linewidth=2, elinewidth=1)
        
        # Plot on separate argmax plot
        ax_argmax.errorbar(fractions_adjusted, mean_argmax, yerr=std_argmax, 
                          label=display_name, 
                          color=color, marker=marker, capsize=5, markersize=8, 
                          linewidth=2, elinewidth=1)
    
    # Configure the combined probability KL plot
    ax1.set_title(f'KL Divergence (Probability) - {dataset}', fontsize=14)
    ax1.set_xlabel('Fraction', fontsize=12)
    ax1.set_ylabel('KL Divergence (Probability)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-10)
    
    # Configure the combined argmax KL plot
    ax2.set_title(f'KL Divergence (Argmax) - {dataset}', fontsize=14)
    ax2.set_xlabel('Fraction', fontsize=12)
    ax2.set_ylabel('KL Divergence (Argmax)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-10)
    
    # Configure the separate probability KL plot
    ax_prob.set_title(f'KL Divergence (Probability) - {dataset}', fontsize=14)
    ax_prob.set_xlabel('Fraction', fontsize=12)
    ax_prob.set_ylabel('KL Divergence (Probability)', fontsize=12)
    ax_prob.legend(fontsize=12)
    ax_prob.grid(True, which="both", ls="-", alpha=0.2)
    ax_prob.set_yscale('log')
    ax_prob.set_ylim(bottom=1e-10)
    
    # Configure the separate argmax KL plot
    ax_argmax.set_title(f'KL Divergence (Argmax) - {dataset}', fontsize=14)
    ax_argmax.set_xlabel('Fraction', fontsize=12)
    ax_argmax.set_ylabel('KL Divergence (Argmax)', fontsize=12)
    ax_argmax.legend(fontsize=12)
    ax_argmax.grid(True, which="both", ls="-", alpha=0.2)
    ax_argmax.set_yscale('log')
    ax_argmax.set_ylim(bottom=1e-10)
    
    # Adjust layout for all plots
    fig_combined.tight_layout()
    fig_prob.tight_layout()
    fig_argmax.tight_layout()
    
    # Save all plots if save_path is provided
    if save_path:
        # Save combined plot
        fig_combined.savefig(f"{save_path}_combined.png", dpi=300, bbox_inches='tight')
        
        # Save separate plots
        fig_prob.savefig(f"{save_path}_probability.png", dpi=300, bbox_inches='tight')
        fig_argmax.savefig(f"{save_path}_argmax.png", dpi=300, bbox_inches='tight')
        
        print(f"Saved plots to {save_path}_combined.png, {save_path}_probability.png, and {save_path}_argmax.png")
    else:
        plt.show()
    
    # Close all figures
    plt.close(fig_combined)
    plt.close(fig_prob)
    plt.close(fig_argmax)


def plot_training_curves(
    stats: dict,
    title: str = 'Training Curves',
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot training curves from calibrator statistics.
    
    Args:
        stats: Dictionary containing training statistics
        title: Title for the plot
        save_path: Optional path to save the plot
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, axes = plt.subplots(1, len(stats), figsize=(5 * len(stats), 4))
    
    if len(stats) == 1:
        axes = [axes]
    
    for i, (metric, values) in enumerate(stats.items()):
        if isinstance(values, list) and len(values) > 1:
            axes[i].plot(values)
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].grid(alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes