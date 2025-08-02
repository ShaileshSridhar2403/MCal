"""Run vision benchmarks using the MCal framework.

This script provides the equivalent functionality to the original get_benchmarks.py
but adapted for the MCal refactor structure. It can be used as a drop-in replacement
for running vision calibration benchmarks.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Optional, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from evaluation.benchmarks import CalibrationBenchmark, process_datasets
from utils.io import load_results, save_results


def load_vision_dataset(
    dataset_name: str,
    data_path: str = "./dataset_store",
    fraction_idx: int = 0
) -> Dict[str, torch.Tensor]:
    """Load vision dataset for calibration benchmarking.
    
    This function loads ablated and clean probabilities for vision datasets.
    You would need to adapt this to your specific data loading requirements.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mri', 'breakhis', 'chexpert')
        data_path: Path to the dataset store
        fraction_idx: Which fraction to load (for testing)
        
    Returns:
        Dictionary with 'ablated_probs' and 'clean_probs' tensors
    """
    # Example implementation - you would need to adapt this to your data structure
    print(f"Loading dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    
    # This is a placeholder - replace with your actual data loading logic
    try:
        # Example paths - adapt to your structure
        ablated_path = f"{data_path}/model_outputs/{dataset_name}/vanilla_fill_0/predictions_augmented_train_{fraction_idx}.npy"
        clean_path = f"{data_path}/model_outputs/{dataset_name}/clean/predictions_clean_{fraction_idx}.npy"
        
        # Try to load the data
        if os.path.exists(ablated_path):
            ablated_data = np.load(ablated_path)
            print(f"Loaded ablated data shape: {ablated_data.shape}")
            
            # For this example, we'll use the same data as clean (you would load actual clean data)
            if os.path.exists(clean_path):
                clean_data = np.load(clean_path)
            else:
                print("Warning: Using ablated data as clean data (for demo purposes)")
                clean_data = ablated_data
            
            # Convert to tensors
            ablated_probs = torch.from_numpy(ablated_data).float()
            clean_probs = torch.from_numpy(clean_data).float()
            
            # If 3D (fractions, samples, classes), take the first fraction for simplicity
            if ablated_probs.dim() == 3:
                ablated_probs = ablated_probs[0]
                clean_probs = clean_probs[0]
            
            return {
                'ablated_probs': ablated_probs,
                'clean_probs': clean_probs
            }
        else:
            print(f"Warning: Data file not found at {ablated_path}")
            # Generate synthetic data for demonstration
            return generate_synthetic_dataset(dataset_name)
            
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_dataset(dataset_name)


def generate_synthetic_dataset(dataset_name: str) -> Dict[str, torch.Tensor]:
    """Generate synthetic dataset for demonstration purposes.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with synthetic 'ablated_probs' and 'clean_probs' tensors
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Define dataset-specific parameters
    dataset_configs = {
        'mri': {'n_samples': 1000, 'n_classes': 4},
        'breakhis': {'n_samples': 1200, 'n_classes': 8},
        'chexpert': {'n_samples': 800, 'n_classes': 5},
        'imagenet': {'n_samples': 2000, 'n_classes': 10}
    }
    
    config = dataset_configs.get(dataset_name, {'n_samples': 1000, 'n_classes': 4})
    n_samples = config['n_samples']
    n_classes = config['n_classes']
    
    print(f"Generating synthetic {dataset_name} dataset: {n_samples} samples, {n_classes} classes")
    
    # Generate clean probabilities (well-calibrated)
    clean_logits = torch.randn(n_samples, n_classes)
    clean_probs = torch.softmax(clean_logits, dim=1)
    
    # Generate ablated probabilities (miscalibrated)
    # Add bias and scale to simulate miscalibration
    bias = torch.randn(n_classes) * 0.5
    scale = torch.exp(torch.randn(n_classes) * 0.3)
    
    ablated_logits = clean_logits * scale.unsqueeze(0) + bias.unsqueeze(0)
    ablated_probs = torch.softmax(ablated_logits, dim=1)
    
    return {
        'ablated_probs': ablated_probs,
        'clean_probs': clean_probs
    }


def run_single_dataset_benchmark(
    dataset_name: str,
    methods: Optional[List[str]] = None,
    n_runs: int = 3,
    device: str = "cuda",
    save_dir: str = "./results",
    data_path: str = "./dataset_store"
) -> Dict[str, Any]:
    """Run benchmark on a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        methods: List of methods to evaluate
        n_runs: Number of runs for statistical significance
        device: Device to use for computation
        save_dir: Directory to save results
        data_path: Path to dataset store
        
    Returns:
        Dictionary containing aggregated results
    """
    print(f"\n{'='*60}")
    print(f"RUNNING SINGLE DATASET BENCHMARK: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    data = load_vision_dataset(dataset_name, data_path)
    
    # Initialize benchmarker
    device_obj = torch.device(device) if isinstance(device, str) else device
    benchmarker = CalibrationBenchmark(device=device_obj, save_dir=save_dir)
    
    # Run benchmark
    results = benchmarker.process_single_dataset(
        dataset_name=dataset_name,
        ablated_probs=data['ablated_probs'],
        clean_probs=data['clean_probs'],
        methods=methods,
        n_runs=n_runs,
        verbose=True
    )
    
    return results


def run_multiple_dataset_benchmark(
    dataset_names: List[str],
    methods: Optional[List[str]] = None,
    n_runs: int = 3,
    device: str = "cuda",
    save_dir: str = "./results",
    data_path: str = "./dataset_store"
) -> Dict[str, Dict[str, Any]]:
    """Run benchmark on multiple datasets.
    
    Args:
        dataset_names: List of dataset names
        methods: List of methods to evaluate
        n_runs: Number of runs for statistical significance
        device: Device to use for computation
        save_dir: Directory to save results
        data_path: Path to dataset store
        
    Returns:
        Dictionary mapping dataset names to aggregated results
    """
    print(f"\n{'='*60}")
    print(f"RUNNING MULTIPLE DATASET BENCHMARK")
    print(f"Datasets: {dataset_names}")
    print(f"Methods: {methods}")
    print(f"Runs: {n_runs}")
    print(f"{'='*60}")
    
    # Load all datasets
    datasets = {}
    for dataset_name in dataset_names:
        datasets[dataset_name] = load_vision_dataset(dataset_name, data_path)
    
    # Initialize benchmarker
    device_obj = torch.device(device) if isinstance(device, str) else device
    benchmarker = CalibrationBenchmark(device=device_obj, save_dir=save_dir)
    
    # Run benchmark
    all_results = benchmarker.process_multiple_datasets(
        datasets=datasets,
        methods=methods,
        n_runs=n_runs,
        verbose=True
    )
    
    return all_results


def recreate_results_from_json(
    dataset_name: str,
    save_dir: str = "./results",
    methods: Optional[List[str]] = None
) -> None:
    """Recreate plots and tables from saved JSON results.
    
    Args:
        dataset_name: Name of the dataset
        save_dir: Directory where results are saved
        methods: Methods to include in recreation
    """
    print(f"\n{'='*60}")
    print(f"RECREATING RESULTS FROM JSON: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmarker = CalibrationBenchmark(device=device, save_dir=save_dir)
    
    try:
        benchmarker.load_and_recreate_results(
            dataset_name=dataset_name,
            methods=methods,
            include_mean_baseline=False
        )
        print(f"Successfully recreated results for {dataset_name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure to run the benchmark first to generate results.")


def main():
    """Main function demonstrating different benchmark usage patterns."""
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./results"
    n_runs = 3  # Reduced for demonstration
    
    # Available methods in MCal framework
    available_methods = [
        'mcal',
        'platt', 
        'temperature',
        'optimized_lambda',
        'expectation_prob',
        'expectation_onehot',
        'neural',
        'logits_sharp'
    ]
    
    # Select methods to test (start with a subset for demonstration)
    methods_to_test = [
        'mcal',
        'platt',
        'temperature',
        'expectation_prob'
    ]
    
    print("MCal Vision Benchmarking System")
    print("="*50)
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    print(f"Available methods: {available_methods}")
    print(f"Testing methods: {methods_to_test}")
    print(f"Number of runs: {n_runs}")
    
    # Example 1: Run benchmark on a single dataset
    print("\n" + "="*50)
    print("EXAMPLE 1: Single Dataset Benchmark")
    print("="*50)
    
    try:
        single_results = run_single_dataset_benchmark(
            dataset_name='mri',
            methods=methods_to_test,
            n_runs=n_runs,
            device=device,
            save_dir=save_dir
        )
        print("Single dataset benchmark completed successfully!")
        
    except Exception as e:
        print(f"Single dataset benchmark failed: {e}")
    
    # Example 2: Run benchmark on multiple datasets
    print("\n" + "="*50)
    print("EXAMPLE 2: Multiple Dataset Benchmark")
    print("="*50)
    
    try:
        dataset_names = ['mri', 'breakhis']  # Start with 2 datasets
        multi_results = run_multiple_dataset_benchmark(
            dataset_names=dataset_names,
            methods=methods_to_test[:2],  # Use fewer methods for multi-dataset
            n_runs=2,  # Fewer runs for demonstration
            device=device,
            save_dir=save_dir
        )
        print("Multiple dataset benchmark completed successfully!")
        
    except Exception as e:
        print(f"Multiple dataset benchmark failed: {e}")
    
    # Example 3: Recreate results from JSON
    print("\n" + "="*50)
    print("EXAMPLE 3: Recreate Results from JSON")
    print("="*50)
    
    try:
        recreate_results_from_json(
            dataset_name='mri',
            save_dir=save_dir,
            methods=methods_to_test
        )
        print("Results recreation completed successfully!")
        
    except Exception as e:
        print(f"Results recreation failed: {e}")
    
    print("\n" + "="*50)
    print("BENCHMARK EXAMPLES COMPLETED")
    print("="*50)
    print(f"Check the '{save_dir}' directory for:")
    print("- JSON results in json/ subdirectory")
    print("- Comparison tables as .txt files")
    print("- Plots in plots/ subdirectory")


if __name__ == "__main__":
    main()