"""Evaluation module for MCal calibration framework."""

# Import benchmarks first (it has better dependency handling)
from .benchmarks import CalibrationBenchmark, process_datasets

# Try to import other modules with graceful degradation
try:
    from .metrics import compute_kl_metrics, compute_calibration_metrics
    from .aggregation import aggregate_results, build_kl_comparison_table
    
    __all__ = [
        'CalibrationBenchmark',
        'process_datasets',
        'compute_kl_metrics',
        'compute_calibration_metrics', 
        'aggregate_results',
        'build_kl_comparison_table'
    ]
except ImportError as e:
    print(f"Warning: Could not import some evaluation components: {e}")
    __all__ = [
        'CalibrationBenchmark',
        'process_datasets'
    ]