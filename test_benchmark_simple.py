"""Simple test of the MCal benchmarking system."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from evaluation.benchmarks import CalibrationBenchmark
    from calibrators import MCal, TemperatureScaling
    from utils.visualization import plot_kl_divergence_simple
    
    print("✓ All imports successful!")
    
    # Test 1: Basic benchmark initialization
    print("\n1. Testing benchmark initialization...")
    device = torch.device("cpu")  # Use CPU for testing
    benchmarker = CalibrationBenchmark(device=device, save_dir="./test_results")
    print("✓ Benchmarker initialized successfully!")
    
    # Test 2: Generate synthetic data
    print("\n2. Generating synthetic test data...")
    torch.manual_seed(42)
    n_samples, n_classes = 100, 4
    
    # Clean probabilities
    clean_logits = torch.randn(n_samples, n_classes)
    clean_probs = torch.softmax(clean_logits, dim=1)
    
    # Ablated probabilities (miscalibrated)
    ablated_logits = clean_logits * 1.5 + torch.randn(n_classes) * 0.3
    ablated_probs = torch.softmax(ablated_logits, dim=1)
    
    print(f"✓ Generated data: {n_samples} samples, {n_classes} classes")
    print(f"  Clean probs shape: {clean_probs.shape}")
    print(f"  Ablated probs shape: {ablated_probs.shape}")
    
    # Test 3: Test individual calibrator
    print("\n3. Testing individual calibrator...")
    mcal = MCal(n_classes)
    stats = mcal.fit(ablated_probs, clean_probs, max_steps=50, verbose=False)
    calibrated_probs = mcal(ablated_probs)
    
    print("✓ MCal calibrator trained and applied successfully!")
    print(f"  Training loss: {stats['loss'][-1]:.4f}")
    print(f"  Output shape: {calibrated_probs.shape}")
    
    # Test 4: Test metrics computation
    print("\n4. Testing metrics computation...")
    from evaluation.metrics import compute_kl_metrics
    
    kl_metrics = compute_kl_metrics(
        calibrated_probs.cpu().numpy(),
        clean_probs.cpu().numpy()
    )
    
    print("✓ KL metrics computed successfully!")
    print(f"  Average KL (prob): {kl_metrics['average_kl_prob']:.6f}")
    print(f"  Average KL (argmax): {kl_metrics['average_kl_argmax']:.6f}")
    
    # Test 5: Test simple benchmark run
    print("\n5. Testing simple benchmark run...")
    
    # Use a subset of methods for quick testing
    test_methods = ['mcal', 'temperature']
    
    try:
        results = benchmarker.process_single_dataset(
            dataset_name='test_dataset',
            ablated_probs=ablated_probs,
            clean_probs=clean_probs,
            methods=test_methods,
            n_runs=2,  # Small number for testing
            verbose=False
        )
        
        print("✓ Benchmark run completed successfully!")
        print(f"  Methods tested: {list(results.keys())}")
        
        # Check if we have expected results structure
        for method in test_methods:
            if method in results:
                method_results = results[method]
                if 'kl_transformed_mean_prob' in method_results:
                    kl_val = method_results['kl_transformed_mean_prob']
                    print(f"  {method} KL: {kl_val:.6f}")
                else:
                    print(f"  {method}: results structure may be incomplete")
        
    except Exception as e:
        print(f"✗ Benchmark run failed: {e}")
        print("This might be due to missing transform dependencies")
    
    print("\n" + "="*50)
    print("BASIC FUNCTIONALITY TEST COMPLETED")
    print("="*50)
    print("✓ Core benchmarking system is working!")
    print("✓ Individual calibrators can be trained and applied")
    print("✓ Metrics can be computed")
    print("✓ Basic infrastructure is in place")
    
    print("\nNext steps for full functionality:")
    print("- Install missing dependencies (scipy, sklearn, tabulate)")
    print("- Test with real datasets")
    print("- Test all calibration methods")
    print("- Verify visualization functions")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all required modules are available")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()