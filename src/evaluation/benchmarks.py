"""Main benchmarking orchestrator for MCal calibration evaluation."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Handle imports gracefully with try/except
try:
    import numpy as np
    import torch
    from tqdm import tqdm
    HAS_TORCH = True
    HAS_NUMPY = True
    HAS_TQDM = True
except ImportError as e:
    print(f"Warning: Missing dependencies: {e}")
    HAS_TORCH = False
    HAS_NUMPY = False
    HAS_TQDM = False

# Try to import MCal components
try:
    # Use sys.path approach instead of relative imports to avoid package issues
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from calibrators import MCal, PlattCalibrator, TemperatureScaling
    from transforms.lambda_transforms import OptimizedLambdaTransform, ExpectationLambdaTransform  
    from transforms.logits import LogitsSharpTransform
    from transforms.neural import NeuralTransform
    from utils.optimization import kl_divergence
    from utils.io import save_results, load_results
    from evaluation.aggregation import aggregate_results, aggregate_fractionwise_kl
    from evaluation.metrics import compute_kl_metrics, compute_calibration_metrics
    from utils.visualization import plot_kl_divergence
    
    HAS_MCAL_COMPONENTS = True
except ImportError as e:
    print(f"Warning: Could not import MCal components: {e}")
    HAS_MCAL_COMPONENTS = False


class CalibrationBenchmark:
    """Main benchmarking orchestrator for calibration methods.
    
    This class provides the equivalent functionality to the original get_benchmarks.py,
    organizing all calibration methods and evaluation in a structured way.
    """
    
    def __init__(
        self,
        device: Optional[Any] = None,  # Changed from torch.device to Any for compatibility
        save_dir: str = "./results",
    ):
        """Initialize the benchmark.
        
        Args:
            device: Device to run computations on
            save_dir: Directory to save results
        """
        if HAS_TORCH and device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        (self.save_dir / "json").mkdir(exist_ok=True)
        
        # Method configurations - only include if components are available
        self.method_configs = {}
        
        if HAS_MCAL_COMPONENTS:
            self.method_configs.update({
                'mcal': {
                    'class': MCal,
                    'params': {'kappa': 10.0, 'max_steps': 10000, 'lr': 1e-2}
                },
                'platt': {
                    'class': PlattCalibrator,
                    'params': {'max_steps': 10000, 'lr': 1e-2}
                },
                'temperature': {
                    'class': TemperatureScaling,
                    'params': {'max_steps': 1000, 'lr': 0.01}
                },
                'optimized_lambda': {
                    'class': OptimizedLambdaTransform,
                    'params': {'batch_size': 256, 'num_epochs': 1000}
                },
                'expectation_prob': {
                    'class': ExpectationLambdaTransform,
                    'params': {'method': 'prob'}
                },
                'expectation_onehot': {
                    'class': ExpectationLambdaTransform,
                    'params': {'method': 'onehot'}
                },
                'logits_sharp': {
                    'class': LogitsSharpTransform,
                    'params': {}
                },
                'neural': {
                    'class': NeuralTransform,
                    'params': {'num_epochs': 100, 'lr': 1e-3}
                }
            })
        else:
            print("Warning: MCal components not available. Limited functionality.")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are available."""
        return {
            'torch': HAS_TORCH,
            'numpy': HAS_NUMPY, 
            'tqdm': HAS_TQDM,
            'mcal_components': HAS_MCAL_COMPONENTS
        }
    
    def process_single_dataset(
        self,
        dataset_name: str,
        ablated_probs: Any,  # Changed from torch.Tensor for compatibility
        clean_probs: Any,    # Changed from torch.Tensor for compatibility  
        methods: Optional[List[str]] = None,
        n_runs: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Process a single dataset with multiple calibration methods.
        
        Args:
            dataset_name: Name of the dataset
            ablated_probs: Ablated probability distributions
            clean_probs: Clean probability distributions  
            methods: List of methods to evaluate
            n_runs: Number of runs for statistical significance
            overwrite: Whether to overwrite existing results
            verbose: Whether to show progress
            
        Returns:
            Dictionary containing aggregated results
        """
        if methods is None:
            methods = list(self.method_configs.keys())
        
        # Initialize results storage
        all_results = {method: [] for method in methods}
        
        if verbose:
            print(f"\nProcessing dataset: {dataset_name}")
            print(f"Data shape: {ablated_probs.shape}")
            print(f"Methods: {methods}")
            print(f"Runs: {n_runs}")
        
        # Run multiple iterations for statistical significance
        for run in tqdm(range(n_runs), desc=f"Processing {dataset_name}"):
            run_results = self._process_single_run(
                ablated_probs, clean_probs, methods, run, verbose=verbose and run == 0
            )
            
            # Store results for each method
            for method, result in run_results.items():
                all_results[method].append(result)
        
        # Aggregate results across runs
        if verbose:
            print(f"\nAggregating results across {n_runs} runs...")
        
        aggregated_results = aggregate_results(all_results)
        
        # Save results
        self._save_dataset_results(dataset_name, aggregated_results, methods, n_runs)
        
        return aggregated_results
    
    def _process_single_run(
        self,
        ablated_probs: Any,  # Changed from torch.Tensor
        clean_probs: Any,    # Changed from torch.Tensor
        methods: List[str],
        run_id: int,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Process a single run with all methods.
        
        Args:
            ablated_probs: Ablated probability distributions
            clean_probs: Clean probability distributions
            methods: List of methods to evaluate
            run_id: Run identifier
            verbose: Whether to show progress
            
        Returns:
            Dictionary containing results for each method
        """
        results = {}
        num_classes = ablated_probs.shape[1]
        
        for method in methods:
            if verbose:
                print(f"  Processing method: {method}")
            
            try:
                # Get method configuration
                config = self.method_configs[method]
                
                # Initialize and fit the method
                if HAS_TORCH and hasattr(torch, 'nn') and issubclass(config['class'], torch.nn.Module):
                    # Calibrator methods (MCal, Platt, Temperature)
                    calibrator = config['class'](num_classes)
                    calibrator.to(self.device)
                    
                    # Fit the calibrator
                    stats = calibrator.fit(
                        ablated_probs.to(self.device),
                        clean_probs.to(self.device),
                        verbose=False,
                        **config['params']
                    )
                    
                    # Get calibrated predictions
                    if HAS_TORCH:
                        with torch.no_grad():
                            calibrated_probs = calibrator(ablated_probs.to(self.device))
                    else:
                        calibrated_probs = calibrator(ablated_probs)
                    
                    # Compute metrics
                    result = self._compute_method_metrics(
                        ablated_probs, clean_probs, calibrated_probs, method, stats
                    )
                
                else:
                    # Transform methods
                    transform = config['class'](device=self.device)
                    
                    # For transforms, we need to save data temporarily
                    temp_path = f"/tmp/mcal_temp_{method}_{run_id}.npy"
                    np.save(temp_path, ablated_probs.cpu().numpy())
                    
                    # Fit the transform
                    stats = transform.fit(temp_path, **config['params'])
                    
                    # Apply transform
                    transformed_probs = transform.transform(ablated_probs.to(self.device))
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    # Compute metrics
                    result = self._compute_method_metrics(
                        ablated_probs, clean_probs, transformed_probs, method, stats
                    )
                
                results[method] = result
                
            except Exception as e:
                print(f"Warning: Method {method} failed with error: {e}")
                # Store empty result to maintain structure
                results[method] = {
                    'kl_results_transformed': {'average_kl_prob': float('inf'), 'average_kl_argmax': float('inf')},
                    'kl_values_transformed': [],
                    'error': str(e)
                }
        
        return results
    
    def _compute_method_metrics(
        self,
        original_probs: Any,  # Changed from torch.Tensor
        target_probs: Any,    # Changed from torch.Tensor
        transformed_probs: Any,  # Changed from torch.Tensor
        method_name: str,
        training_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute comprehensive metrics for a method.
        
        Args:
            original_probs: Original probability distributions
            target_probs: Target probability distributions
            transformed_probs: Transformed probability distributions
            method_name: Name of the method
            training_stats: Training statistics from the method
            
        Returns:
            Dictionary containing all computed metrics
        """
        # Move tensors to CPU for numpy operations (if they are tensors)
        if HAS_TORCH and hasattr(original_probs, 'cpu'):
            original_np = original_probs.cpu().numpy()
            target_np = target_probs.cpu().numpy()
            transformed_np = transformed_probs.cpu().numpy()
        else:
            # Assume they are already numpy arrays or convert
            original_np = np.array(original_probs) if HAS_NUMPY else original_probs
            target_np = np.array(target_probs) if HAS_NUMPY else target_probs
            transformed_np = np.array(transformed_probs) if HAS_NUMPY else transformed_probs
        
        # Compute KL divergence metrics (if numpy is available)
        if HAS_NUMPY and HAS_MCAL_COMPONENTS:
            kl_metrics = compute_kl_metrics(transformed_np, target_np)
            calibration_metrics = compute_calibration_metrics(transformed_np, target_np)
        else:
            kl_metrics = {'average_kl_prob': 0.0, 'average_kl_argmax': 0.0, 'fractionwise_kl': []}
            calibration_metrics = {}
        
        # Package results in the format expected by the original system
        result = {
            'method': method_name,
            'kl_results_transformed': {
                'average_kl_prob': kl_metrics['average_kl_prob'],
                'average_kl_argmax': kl_metrics['average_kl_argmax']
            },
            'kl_values_transformed': kl_metrics['fractionwise_kl'],
            'calibration_metrics': calibration_metrics,
            'training_stats': training_stats
        }
        
        # Add baseline metrics for comparison (using original probabilities)
        if method_name in ['mcal', 'platt', 'temperature']:  # Only for first method to avoid duplication
            baseline_kl_metrics = compute_kl_metrics(original_np, target_np)
            result['kl_results_baseline'] = {
                'average_kl_prob': baseline_kl_metrics['average_kl_prob'],
                'average_kl_argmax': baseline_kl_metrics['average_kl_argmax']
            }
            result['kl_values_baseline'] = baseline_kl_metrics['fractionwise_kl']
        
        return result
    
    def _save_dataset_results(
        self,
        dataset_name: str,
        aggregated_results: Dict[str, Any],
        methods: List[str],
        n_runs: int
    ) -> None:
        """Save dataset results and generate visualizations.
        
        Args:
            dataset_name: Name of the dataset
            aggregated_results: Aggregated results dictionary
            methods: List of methods used
            n_runs: Number of runs performed
        """
        # Save aggregated results as JSON
        json_path = self.save_dir / "json" / f"aggregated_results_{dataset_name}.json"
        save_results(aggregated_results, str(json_path))
        
        # Generate comparison table
        from .aggregation import build_kl_comparison_table
        table = build_kl_comparison_table(aggregated_results, include_methods=methods)
        
        # Save table
        table_path = self.save_dir / f"kl_comparison_table_{dataset_name}.txt"
        with open(table_path, 'w') as f:
            f.write(f"KL Divergence Comparison for {dataset_name} (averaged over {n_runs} runs):\n")
            f.write(table)
        
        # Generate plots
        plot_path = self.save_dir / "plots" / f"kl_divergence_plot_{dataset_name}"
        plot_kl_divergence(
            aggregated_results,
            dataset_name,
            save_path=str(plot_path),
            include_methods=methods
        )
        
        print(f"Results saved:")
        print(f"  JSON: {json_path}")
        print(f"  Table: {table_path}")
        print(f"  Plots: {plot_path}_*.png")
    
    def process_multiple_datasets(
        self,
        datasets: Dict[str, Dict[str, Any]],  # Changed from torch.Tensor
        methods: Optional[List[str]] = None,
        n_runs: int = 5,
        overwrite: bool = False,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Process multiple datasets with benchmarking.
        
        Args:
            datasets: Dictionary mapping dataset names to data dictionaries
                     Each data dict should have 'ablated_probs' and 'clean_probs' keys
            methods: List of methods to evaluate
            n_runs: Number of runs for statistical significance
            overwrite: Whether to overwrite existing results
            verbose: Whether to show progress
            
        Returns:
            Dictionary mapping dataset names to aggregated results
        """
        all_dataset_results = {}
        
        for dataset_name, data in datasets.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"PROCESSING DATASET: {dataset_name.upper()}")
                print(f"{'='*60}")
            
            results = self.process_single_dataset(
                dataset_name=dataset_name,
                ablated_probs=data['ablated_probs'],
                clean_probs=data['clean_probs'],
                methods=methods,
                n_runs=n_runs,
                overwrite=overwrite,
                verbose=verbose
            )
            
            all_dataset_results[dataset_name] = results
        
        # Generate cross-dataset summary
        if len(datasets) > 1:
            self._generate_cross_dataset_summary(all_dataset_results, methods, n_runs)
        
        return all_dataset_results
    
    def _generate_cross_dataset_summary(
        self,
        all_results: Dict[str, Dict[str, Any]],
        methods: List[str],
        n_runs: int
    ) -> None:
        """Generate summary across all datasets.
        
        Args:
            all_results: Results for all datasets
            methods: Methods used
            n_runs: Number of runs
        """
        print(f"\n{'='*60}")
        print("CROSS-DATASET SUMMARY")
        print(f"{'='*60}")
        
        # Create summary table
        from .aggregation import build_cross_dataset_summary_table
        summary_table = build_cross_dataset_summary_table(all_results, methods)
        
        # Save summary
        summary_path = self.save_dir / "cross_dataset_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Cross-Dataset Summary (averaged over {n_runs} runs per dataset):\n\n")
            f.write(summary_table)
        
        print(f"Cross-dataset summary saved to: {summary_path}")
        
        # Save combined JSON
        combined_json_path = self.save_dir / "json" / "all_datasets_results.json"
        save_results(all_results, str(combined_json_path))
        print(f"Combined results saved to: {combined_json_path}")
    
    def load_and_recreate_results(
        self,
        dataset_name: str,
        methods: Optional[List[str]] = None,
        include_mean_baseline: bool = False
    ) -> None:
        """Load saved results and recreate plots/tables.
        
        Args:
            dataset_name: Name of the dataset
            methods: Methods to include in recreation
            include_mean_baseline: Whether to include mean baseline
        """
        # Load results
        json_path = self.save_dir / "json" / f"aggregated_results_{dataset_name}.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Results file not found: {json_path}")
        
        aggregated_results = load_results(str(json_path))
        
        # Recreate table
        from .aggregation import build_kl_comparison_table
        table = build_kl_comparison_table(
            aggregated_results, 
            include_methods=methods,
            include_mean_baseline=include_mean_baseline
        )
        
        print(f"Recreated table for {dataset_name}:")
        print(table)
        
        # Recreate plots
        plot_path = self.save_dir / "plots" / f"kl_divergence_plot_{dataset_name}_recreated"
        plot_kl_divergence(
            aggregated_results,
            dataset_name,
            save_path=str(plot_path),
            include_methods=methods,
            include_mean_baseline=include_mean_baseline
        )
        
        print(f"Recreated plots saved to: {plot_path}_*.png")
    
    def get_method_list(self) -> List[str]:
        """Get list of available methods."""
        return list(self.method_configs.keys())
    
    def add_custom_method(
        self,
        method_name: str,
        method_class: type,
        method_params: Dict[str, Any]
    ) -> None:
        """Add a custom method to the benchmark.
        
        Args:
            method_name: Name for the method
            method_class: Class implementing the method
            method_params: Parameters for the method
        """
        self.method_configs[method_name] = {
            'class': method_class,
            'params': method_params
        }
        print(f"Added custom method: {method_name}")


# Convenience functions that replicate the original get_benchmarks.py interface
def process_datasets(
    dataset_types: Optional[List[str]] = None,
    device: str = "cuda",
    save_dir: str = "./results",
    n: int = 5,
    methods: Optional[List[str]] = None,
    overwrite: bool = False,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """Process datasets - equivalent to the original process_datasets function.
    
    This function provides the same interface as the original get_benchmarks.py
    but adapted for the MCal refactor structure.
    
    Args:
        dataset_types: List of dataset types to process
        device: Device to use for computation
        save_dir: Directory to save results
        n: Number of iterations for each dataset
        methods: List of methods to use
        overwrite: Whether to overwrite existing results
        **kwargs: Additional arguments
        
    Returns:
        Dictionary mapping dataset names to results
    """
    # Initialize benchmarker
    if HAS_TORCH:
        device_obj = torch.device(device) if isinstance(device, str) else device
    else:
        device_obj = device
    benchmarker = CalibrationBenchmark(device=device_obj, save_dir=save_dir)
    
    # Load datasets (this would need to be implemented based on your data loading logic)
    # For now, we assume datasets are provided or loaded elsewhere
    datasets = {}
    
    if dataset_types is None:
        dataset_types = ["mri", "breakhis", "chexpert", "imagenet"]
    
    # This is where you would load your actual datasets
    # For the refactor, this would need to be connected to your data loading logic
    print("Note: Dataset loading needs to be implemented based on your data structure")
    print(f"Expected datasets: {dataset_types}")
    
    # Process datasets
    return benchmarker.process_multiple_datasets(
        datasets=datasets,
        methods=methods,
        n_runs=n,
        overwrite=overwrite,
        verbose=True
    )