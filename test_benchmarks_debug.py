"""Debug the MCal benchmarking system structure."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_benchmarks_import():
    """Test importing the benchmarks module."""
    try:
        print("Testing benchmarks import...")
        from evaluation.benchmarks import CalibrationBenchmark
        print("✓ CalibrationBenchmark imported successfully!")
        
        # Test initialization
        benchmarker = CalibrationBenchmark(device=None, save_dir="./test_results")
        print("✓ CalibrationBenchmark initialized successfully!")
        
        # Check dependencies
        deps = benchmarker.check_dependencies()
        print(f"\nDependency status:")
        for dep, available in deps.items():
            status = "✓" if available else "✗"
            print(f"  {status} {dep}: {available}")
        
        # Check available methods
        methods = list(benchmarker.method_configs.keys())
        print(f"\nAvailable methods: {methods}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_other_modules():
    """Test importing other evaluation modules."""
    modules_to_test = [
        ("evaluation.metrics", "Metrics module"),
        ("evaluation.aggregation", "Aggregation module"),
        ("utils.visualization", "Visualization module"),
        ("utils.io", "I/O module")
    ]
    
    success_count = 0
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {description}")
            success_count += 1
        except Exception as e:
            print(f"✗ {description}: {e}")
    
    return success_count, len(modules_to_test)

def main():
    print("MCal Benchmarking System - Debug Test")
    print("="*50)
    
    # Test main benchmarks module
    benchmarks_ok = test_benchmarks_import()
    
    print(f"\n{'='*50}")
    print("Testing other modules...")
    print("="*50)
    
    # Test other modules
    success, total = test_other_modules()
    
    print(f"\n{'='*50}")
    print("DEBUG SUMMARY")
    print("="*50)
    
    if benchmarks_ok:
        print("✓ Main benchmarking system structure is working!")
        print("✓ Can initialize CalibrationBenchmark")
        print("✓ Dependency checking works")
    else:
        print("✗ Main benchmarking system has issues")
    
    print(f"✓ {success}/{total} supporting modules working")
    
    print(f"\n{'='*50}")
    print("DIAGNOSIS")
    print("="*50)
    
    if benchmarks_ok:
        print("✓ The structure is correct and imports work!")
        print("✓ The issue was likely the relative import problem")
        print("✓ Fixed by using sys.path approach")
        print("\nNext steps:")
        print("- Install dependencies (torch, numpy, scipy, sklearn)")
        print("- Test with actual data")
        print("- Run full benchmarking pipeline")
    else:
        print("✗ Still have structural issues to resolve")
        print("Check the error messages above for details")

if __name__ == "__main__":
    main()