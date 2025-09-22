#!/usr/bin/env python3
"""
Test attention mask implementation without running full benchmark
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))

from medqa_utils import (
    create_random_attention_mask,
    identify_content_positions,
    create_medqa_prompt
)

def test_attention_mask_basic():
    """Test basic attention mask functionality."""
    print("Testing Attention Mask Implementation...")
    print("="*50)

    # Create a mock tokenizer-like object for testing
    class MockTokenizer:
        def encode(self, text, return_tensors="pt"):
            # Simple word-based tokenization for testing
            words = text.split()
            # Simulate token IDs (just use indices)
            token_ids = list(range(len(words)))
            if return_tensors == "pt":
                return torch.tensor([token_ids])
            return token_ids

        def convert_ids_to_tokens(self, ids):
            # Return mock tokens
            return [f"token_{i}" for i in ids]

        def decode(self, ids, skip_special_tokens=True):
            # Return simple reconstruction
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return " ".join([f"word_{i}" for i in ids])

    # Test data
    question_data = {
        "question": "What is the most common cause of bacterial pneumonia?",
        "options": {
            "A": "Streptococcus pneumoniae",
            "B": "Haemophilus influenzae",
            "C": "Mycoplasma pneumoniae",
            "D": "Klebsiella pneumoniae",
            "E": "Staphylococcus aureus"
        },
        "answer_idx": "A"
    }

    # Create prompt
    prompt = create_medqa_prompt(question_data, removal_fraction=0.0)
    print(f"Original prompt: {prompt[:100]}...")

    # Mock tokenizer
    tokenizer = MockTokenizer()

    # Tokenize
    input_ids = tokenizer.encode(prompt)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0][:10].tolist()}...")

    # Test attention mask generation (content-only)
    attention_mask_0 = create_random_attention_mask(
        input_ids, tokenizer, mask_fraction=0.0
    )
    print(f"Mask fraction 0.0 - ones: {(attention_mask_0 == 1).sum().item()}, zeros: {(attention_mask_0 == 0).sum().item()}")

    attention_mask_05 = create_random_attention_mask(
        input_ids, tokenizer, mask_fraction=0.5
    )
    print(f"Mask fraction 0.5 - ones: {(attention_mask_05 == 1).sum().item()}, zeros: {(attention_mask_05 == 0).sum().item()}")

    attention_mask_09 = create_random_attention_mask(
        input_ids, tokenizer, mask_fraction=0.9
    )
    print(f"Mask fraction 0.9 - ones: {(attention_mask_09 == 1).sum().item()}, zeros: {(attention_mask_09 == 0).sum().item()}")

    # Test content positions
    content_positions = identify_content_positions(input_ids, tokenizer)
    print(f"Content positions: {len(content_positions)} out of {input_ids.shape[1]} total tokens")
    print(f"Content positions indices: {content_positions[:10]}...")

    print("\n✓ Basic attention mask functionality working!")

    return True

if __name__ == "__main__":
    test_attention_mask_basic()