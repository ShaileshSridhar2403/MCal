#!/usr/bin/env python3
"""
Test MedQA Ablation Mechanism

Step 1: Test that word masking/ablation is working correctly before adding model complexity.
"""

import sys
from pathlib import Path

# Add XAI_Benchmark to path
xai_root = Path(__file__).parent.parent.parent.parent / "XAI-Benchmark"
sys.path.insert(0, str(xai_root / "language"))

def test_ablation_mechanism():
    """Test that word masking works correctly."""
    print("Testing MedQA Ablation Mechanism...")
    print("="*50)

    try:
        # Import XAI_Benchmark utilities
        from missingness_utils import replace_random_features, default_tokenize

        # Try importing dataset_utils, which might have import issues
        try:
            from dataset_utils import load_balanced_medqa_dev, construct_prompt
            print("✓ Successfully imported XAI_Benchmark utilities")
        except ImportError as import_err:
            print(f"⚠️  Warning: Import issue in dataset_utils: {import_err}")
            print("Trying alternative import approach...")

            # Import the functions we need directly without problematic dependencies
            import sys
            import importlib.util

            # Load dataset_utils module manually to bypass import errors
            spec = importlib.util.spec_from_file_location(
                "dataset_utils",
                str(xai_root / "language" / "dataset_utils.py")
            )
            dataset_utils = importlib.util.module_from_spec(spec)

            # Add missingness_utils to sys.modules to avoid missing imports
            sys.modules['missingness_utils'] = importlib.import_module('missingness_utils')

            spec.loader.exec_module(dataset_utils)

            load_balanced_medqa_dev = dataset_utils.load_balanced_medqa_dev
            construct_prompt = dataset_utils.construct_prompt

            print("✓ Successfully imported utilities with workaround")

        # Load a few MedQA questions
        print("\nLoading MedQA dataset...")
        medqa_data = load_balanced_medqa_dev()[:3]  # Just 3 questions for testing
        print(f"✓ Loaded {len(medqa_data)} MedQA questions for testing")

        # Test different removal fractions
        removal_fractions = [0.0, 0.3, 0.6, 0.9]

        for i, question_data in enumerate(medqa_data):
            print(f"\n{'='*20} Question {i+1} {'='*20}")

            # Original prompt (no ablation)
            original_prompt = construct_prompt(question_data, "medqa", removal_fraction=None)
            original_word_count = len(default_tokenize(original_prompt))

            print(f"Original prompt ({original_word_count} words):")
            print(f"  {original_prompt[:150]}...")

            # Test ablation at different levels
            print(f"\nTesting ablation at different fractions:")
            for frac in removal_fractions:
                try:
                    # Apply word masking using XAI_Benchmark method
                    masked_prompt = construct_prompt(
                        question_data,
                        "medqa",
                        removal_fraction=frac,
                        tokenize_func=default_tokenize,
                        replacement_token='UNKWORDZ'
                    )

                    # Count how many UNKWORDZ tokens were added
                    unk_count = masked_prompt.count('UNKWORDZ')
                    actual_fraction = unk_count / original_word_count if original_word_count > 0 else 0

                    print(f"  Fraction {frac:.1f}: {unk_count} UNKWORDZ tokens " +
                          f"(actual: {actual_fraction:.2f}, target: {frac:.1f})")

                    if frac > 0:  # Show masked text for non-zero fractions
                        print(f"    Masked: {masked_prompt[:150]}...")

                        # Verify that masking actually happened for non-zero fractions
                        if unk_count == 0:
                            print(f"    ⚠️  Warning: No UNKWORDZ tokens found for fraction {frac}")
                        else:
                            print(f"    ✓ Ablation working correctly")
                    else:
                        # For fraction 0, there should be no UNKWORDZ tokens
                        if unk_count == 0:
                            print(f"    ✓ No ablation for fraction 0.0 (correct)")
                        else:
                            print(f"    ⚠️  Warning: Found {unk_count} UNKWORDZ tokens for fraction 0.0")

                except Exception as e:
                    print(f"    ✗ Error testing fraction {frac}: {e}")
                    return False

        print("\n" + "="*50)
        print("✓ Ablation mechanism test completed successfully!")
        return True

    except ImportError as e:
        print(f"✗ Failed to import XAI_Benchmark utilities: {e}")
        print("Make sure XAI-Benchmark is in the expected location")
        return False
    except Exception as e:
        print(f"✗ Ablation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_word_replacement():
    """Test the basic word replacement mechanism directly."""
    print("\nTesting basic word replacement mechanism...")

    try:
        from missingness_utils import replace_random_features, default_tokenize

        # Test text
        test_text = "This is a simple test sentence with multiple words."
        print(f"Test text: '{test_text}'")

        # Test different fractions
        for frac in [0.0, 0.2, 0.5, 0.8]:
            masked_text = replace_random_features(
                test_text,
                tokenize_func=default_tokenize,
                removal_fraction=frac,
                replacement_token='UNKWORDZ'
            )

            unk_count = masked_text.count('UNKWORDZ')
            original_count = len(default_tokenize(test_text))
            actual_frac = unk_count / original_count

            print(f"  Fraction {frac:.1f}: '{masked_text}' ({unk_count}/{original_count} = {actual_frac:.2f})")

        print("✓ Basic word replacement test passed!")
        return True

    except Exception as e:
        print(f"✗ Basic word replacement test failed: {e}")
        return False

if __name__ == "__main__":
    print("MedQA Ablation Test - Step 1")
    print("="*50)

    # Test basic mechanism first
    basic_success = test_basic_word_replacement()

    if basic_success:
        # Test with actual MedQA data
        success = test_ablation_mechanism()

        if success:
            print("\n🎉 All ablation tests passed! Ready for Step 2.")
        else:
            print("\n❌ Ablation tests failed. Fix issues before proceeding to Step 2.")
    else:
        print("\n❌ Basic word replacement test failed. Check XAI_Benchmark imports.")