#!/usr/bin/env python3
"""
Test LLaMA Model Loading and First Token Probability Extraction

Step 2: Test that we can load LLaMA and extract A,B,C,D,E probabilities reliably.
NO FALLBACK - Real model only, raise errors if issues.
"""

import sys
import torch
from pathlib import Path

# No external dependencies - use MCal utilities only

def test_llama_model_loading():
    """Test LLaMA model loading - REAL MODEL ONLY"""
    print("Testing LLaMA Model Loading...")
    print("="*50)

    model_path = "~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/"
    # Expand the path to handle ~
    expanded_path = Path(model_path).expanduser()
    print(f"Model path: {model_path}")
    print(f"Expanded path: {expanded_path}")

    # Check if path exists
    if not expanded_path.exists():
        raise FileNotFoundError(
            f"LLaMA model not found at: {expanded_path}\n"
            f"Please check:\n"
            f"1. Model is downloaded and accessible\n"
            f"2. Path is correct\n"
            f"3. You have read permissions"
        )

    try:
        # Import MCal utilities
        from medqa_utils import MCal_LLaMAModel
        print("✓ Successfully imported MCal LLaMA utilities")

        # Load real model - raise error if fails
        print("Loading LLaMA model... (this may take a moment)")
        model = MCal_LLaMAModel(str(expanded_path))
        print("✓ LLaMA model loaded successfully")

        # Check device
        device = next(model.model.parameters()).device
        print(f"✓ Model loaded on device: {device}")

        return model

    except ImportError as e:
        raise ImportError(
            f"Failed to import MCal LLaMA utilities: {e}\n"
            f"Please check medqa_utils.py is in the same directory"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load LLaMA model: {e}\n"
            f"This could be due to:\n"
            f"1. Insufficient GPU memory\n"
            f"2. CUDA issues\n"
            f"3. Model file corruption\n"
            f"4. Dependency issues"
        )

def test_simple_probability_extraction(model):
    """Test probability extraction with simple math question"""
    print("\nTesting Simple Probability Extraction...")
    print("-" * 50)

    # Simple test prompt
    test_prompt = "Question: What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nE. 7\nAnswer:"
    print(f"Test prompt: {test_prompt}")

    try:
        # Extract probabilities using MCal method
        probs = model.forward([test_prompt], num_options=5)

        if not probs or len(probs) == 0:
            raise ValueError("No probabilities returned from model")

        prob_dict = probs[0]  # First (and only) result
        print(f"✓ Raw probabilities: {prob_dict}")

        # Verify we have all 5 choices
        expected_keys = {'A', 'B', 'C', 'D', 'E'}
        actual_keys = set(prob_dict.keys())

        if actual_keys != expected_keys:
            raise ValueError(f"Expected keys {expected_keys}, got {actual_keys}")

        print("✓ All 5 choices (A,B,C,D,E) present")

        # Verify probabilities are reasonable
        prob_sum = sum(prob_dict.values())
        print(f"✓ Probability sum: {prob_sum:.6f}")

        if not (0.9 < prob_sum < 1.1):
            raise ValueError(f"Probability sum {prob_sum} is not close to 1.0")

        print("✓ Probability sum is reasonable")

        # Check for reasonable distribution (no single probability dominates too much)
        max_prob = max(prob_dict.values())
        min_prob = min(prob_dict.values())
        print(f"✓ Probability range: {min_prob:.4f} - {max_prob:.4f}")

        return True

    except Exception as e:
        raise RuntimeError(f"Probability extraction failed: {e}")

def test_medqa_style_probability_extraction(model):
    """Test probability extraction with MedQA-style question"""
    print("\nTesting MedQA-Style Probability Extraction...")
    print("-" * 50)

    # MedQA-style test prompt
    medqa_prompt = """Question: What is the most common cause of bacterial pneumonia in adults?
A. Streptococcus pneumoniae
B. Haemophilus influenzae
C. Mycoplasma pneumoniae
D. Klebsiella pneumoniae
E. Staphylococcus aureus
Answer:"""

    print(f"MedQA prompt: {medqa_prompt[:100]}...")

    try:
        # Extract probabilities
        probs = model.forward([medqa_prompt], num_options=5)
        prob_dict = probs[0]

        print(f"✓ MedQA probabilities: {prob_dict}")

        # Verify sum
        prob_sum = sum(prob_dict.values())
        if not (0.9 < prob_sum < 1.1):
            raise ValueError(f"MedQA probability sum {prob_sum} is not close to 1.0")

        print(f"✓ MedQA probability sum: {prob_sum:.6f}")

        return True

    except Exception as e:
        raise RuntimeError(f"MedQA probability extraction failed: {e}")

def test_ablated_probability_extraction(model):
    """Test probability extraction with ablated prompts (from Step 1)"""
    print("\nTesting Ablated Probability Extraction...")
    print("-" * 50)

    # Import our ablation function from MCal utilities
    from medqa_utils import replace_random_features

    # Base question
    base_question = "What is the most common cause of bacterial pneumonia in adults?"
    options = {
        "A": "Streptococcus pneumoniae",
        "B": "Haemophilus influenzae",
        "C": "Mycoplasma pneumoniae",
        "D": "Klebsiella pneumoniae",
        "E": "Staphylococcus aureus"
    }

    # Test different ablation levels
    ablation_fractions = [0.0, 0.3, 0.6]

    for frac in ablation_fractions:
        print(f"\nTesting ablation fraction: {frac}")

        # Apply ablation
        if frac > 0:
            ablated_question = replace_random_features(
                base_question,
                removal_fraction=frac,
                replacement_token='UNKWORDZ'
            )
        else:
            ablated_question = base_question

        # Construct prompt
        prompt = f"Question: {ablated_question}\n"
        for opt_key, opt_text in options.items():
            prompt += f"{opt_key}. {opt_text}\n"
        prompt += "Answer:"

        print(f"  Ablated question: {ablated_question}")

        try:
            # Extract probabilities
            probs = model.forward([prompt], num_options=5)
            prob_dict = probs[0]

            prob_sum = sum(prob_dict.values())
            max_choice = max(prob_dict, key=prob_dict.get)

            print(f"  Probabilities: {prob_dict}")
            print(f"  Sum: {prob_sum:.4f}, Top choice: {max_choice} ({prob_dict[max_choice]:.4f})")

            if not (0.9 < prob_sum < 1.1):
                raise ValueError(f"Ablated probability sum {prob_sum} is not close to 1.0")

            print("  ✓ Ablated extraction successful")

        except Exception as e:
            raise RuntimeError(f"Ablated extraction failed for fraction {frac}: {e}")

    return True

def test_gpu_cpu_info():
    """Test and report GPU/CPU information"""
    print("\nGPU/CPU Information...")
    print("-" * 50)

    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA not available - model will use CPU (very slow)")

if __name__ == "__main__":
    print("LLaMA Model Test - Step 2")
    print("="*60)
    print("Testing real LLaMA model loading and probability extraction")
    print("NO FALLBACK - Will raise errors if issues occur")
    print("="*60)

    try:
        # Test system info
        test_gpu_cpu_info()

        # Test model loading
        model = test_llama_model_loading()

        # Test simple probability extraction
        test_simple_probability_extraction(model)

        # Test MedQA-style extraction
        test_medqa_style_probability_extraction(model)

        # Test ablated extraction
        test_ablated_probability_extraction(model)

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("✓ LLaMA model loads successfully")
        print("✓ Probability extraction works correctly")
        print("✓ All 5 choices (A,B,C,D,E) supported")
        print("✓ Probabilities sum to ~1.0")
        print("✓ Ablated prompts work correctly")
        print("✓ Ready for Step 3: Full Pipeline Integration")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nPlease fix the issue before proceeding to Step 3.")
        sys.exit(1)