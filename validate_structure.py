"""Validate the MCal directory structure and imports."""

import os
import sys
from pathlib import Path

def check_file_exists(path, description=""):
    """Check if a file exists and print status."""
    if os.path.exists(path):
        print(f"✓ {path} {description}")
        return True
    else:
        print(f"✗ {path} {description}")
        return False

def validate_structure():
    """Validate the MCal directory structure."""
    print("Validating MCal directory structure...")
    print("=" * 50)
    
    base_path = Path(__file__).parent
    
    # Core structure
    structure_checks = [
        ("README.md", "Main README"),
        ("setup.py", "Package setup"),
        ("src/", "Source directory"),
        ("src/__init__.py", "Main package init"),
        ("src/calibrators/", "Calibrators directory"),
        ("src/calibrators/__init__.py", "Calibrators init"),
        ("src/calibrators/base.py", "Base calibrator"),
        ("src/calibrators/mcal.py", "MCal calibrator"),
        ("src/calibrators/platt.py", "Platt calibrator"),
        ("src/calibrators/temperature.py", "Temperature scaling"),
        ("src/transforms/", "Transforms directory"),
        ("src/transforms/__init__.py", "Transforms init"),
        ("src/transforms/base.py", "Base transform"),
        ("src/transforms/lambda_transforms.py", "Lambda transforms"),
        ("src/transforms/calibration.py", "Calibration transforms"),
        ("src/transforms/neural.py", "Neural transforms"),
        ("src/transforms/logits.py", "Logit transforms"),
        ("src/utils/", "Utils directory"),
        ("src/utils/__init__.py", "Utils init"),
        ("src/utils/optimization.py", "Optimization utils"),
        ("src/utils/io.py", "IO utils"),
        ("src/utils/visualization.py", "Visualization utils"),
        ("tests/", "Tests directory"),
        ("tests/test_calibrators.py", "Calibrator tests"),
        ("experiments/", "Experiments directory"),
        ("configs/", "Configs directory"),
        ("finetune/", "Finetune directory"),
        ("notebooks/", "Notebooks directory"),
        ("notebooks/calibration_demo.ipynb", "Demo notebook"),
    ]
    
    passed = 0
    total = len(structure_checks)
    
    for file_path, description in structure_checks:
        full_path = base_path / file_path
        if check_file_exists(full_path, description):
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Structure validation: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ All structure checks passed!")
        return True
    else:
        print(f"✗ {total - passed} checks failed")
        return False

def validate_imports():
    """Validate that key imports work (if dependencies are available)."""
    print("\nValidating imports...")
    print("=" * 50)
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    import_checks = [
        ("calibrators", "Calibrators module"),
        ("transforms", "Transforms module"),
        ("utils", "Utils module"),
    ]
    
    passed = 0
    total = len(import_checks)
    
    for module_name, description in import_checks:
        try:
            __import__(module_name)
            print(f"✓ {module_name} - {description}")
            passed += 1
        except ImportError as e:
            print(f"✗ {module_name} - {description} (Error: {e})")
        except Exception as e:
            print(f"⚠ {module_name} - {description} (Warning: {e})")
    
    print(f"\nImport validation: {passed}/{total} checks passed")
    return passed == total

def main():
    """Main validation function."""
    print("MCal Framework Validation")
    print("=" * 50)
    
    structure_ok = validate_structure()
    imports_ok = validate_imports()
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    if structure_ok and imports_ok:
        print("✓ MCal framework validation PASSED!")
        print("The refactor was successful.")
        return True
    else:
        if not structure_ok:
            print("✗ Structure validation failed")
        if not imports_ok:
            print("⚠ Import validation had issues (may need dependencies)")
        print("Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)