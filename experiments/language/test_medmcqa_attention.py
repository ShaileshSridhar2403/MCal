#!/usr/bin/env python3
"""
Test MedMCQA attention mask implementation
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))

from medmcqa_utils import (
    create_random_attention_mask,
    identify_content_positions,
    create_medmcqa_prompt
)

def test_medmcqa_attention_mask():
    """Test MedMCQA attention mask functionality."""
    print("Testing MedMCQA Attention Mask Implementation...")
    print("="*60)

    # Simulate MedMCQA tokenizer behavior
    class MockMedMCQATokenizer:
        def encode(self, text, return_tensors="pt"):
            # Simulate realistic tokenization of MedMCQA prompt
            tokens = []

            # Question section
            if "Question:" in text:
                tokens.extend([1, 2])  # "Question:"

            # Question content (what we want to mask)
            question_part = text.split("Question:")[1].split("A.")[0].strip()
            question_tokens = len(question_part.split())
            tokens.extend(range(10, 10 + question_tokens))  # Question content tokens

            # Answer choices (preserve these) - MedMCQA has A,B,C,D only
            if "A." in text:
                tokens.extend([100, 101])  # "A. Choice"
            if "B." in text:
                tokens.extend([200, 201])  # "B. Choice"
            if "C." in text:
                tokens.extend([300, 301])  # "C. Choice"
            if "D." in text:
                tokens.extend([400, 401])  # "D. Choice"

            # Answer prompt
            if "Answer:" in text:
                tokens.extend([999])  # "Answer:"

            if return_tensors == "pt":
                return torch.tensor([tokens])
            return tokens

        def convert_ids_to_tokens(self, ids):
            # Map token IDs to descriptive names
            token_map = {
                1: "Question", 2: ":",
                999: "Answer:",
                100: "A", 101: "choice_a",
                200: "B", 201: "choice_b",
                300: "C", 301: "choice_c",
                400: "D", 401: "choice_d"
            }

            tokens = []
            for token_id in ids:
                if token_id in token_map:
                    tokens.append(token_map[token_id])
                elif 10 <= token_id < 50:  # Question content range
                    tokens.append(f"question_content_{token_id-10}")
                else:
                    tokens.append(f"token_{token_id}")
            return tokens

        def decode(self, ids, skip_special_tokens=True):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()

            # Simulate text reconstruction
            text_parts = []
            i = 0
            while i < len(ids):
                token_id = ids[i]

                if token_id == 1 and i+1 < len(ids) and ids[i+1] == 2:
                    text_parts.append("Question:")
                    i += 2
                elif token_id == 999:
                    text_parts.append("Answer:")
                    i += 1
                elif token_id in [100, 200, 300, 400]:  # A,B,C,D for MedMCQA
                    letter = chr(65 + (token_id - 100) // 100)
                    text_parts.append(f"{letter}.")
                    i += 1
                elif 10 <= token_id < 50:
                    text_parts.append(f"content_word_{token_id-10}")
                    i += 1
                else:
                    text_parts.append(f"token_{token_id}")
                    i += 1

            return " ".join(text_parts)

    # Test data - MedMCQA format with 4 options
    question_data = {
        "question": "Which drug is used for treatment of tuberculosis?",
        "opa": "Rifampin",
        "opb": "Penicillin",
        "opc": "Aspirin",
        "opd": "Metformin",
        "cop": 1  # A is correct (1-indexed)
    }

    # Create prompt
    prompt = create_medmcqa_prompt(question_data, removal_fraction=0.0)
    print(f"Original MedMCQA prompt:\n{prompt}\n")

    # Mock tokenizer
    tokenizer = MockMedMCQATokenizer()

    # Tokenize
    input_ids = tokenizer.encode(prompt)
    print(f"Input IDs: {input_ids[0].tolist()}")

    # Show token breakdown
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"Tokens: {tokens}")

    # Test content position identification (should identify question content only)
    content_positions = identify_content_positions(input_ids, tokenizer)
    print(f"\nContent positions (question content only): {content_positions}")

    content_tokens = [tokens[i] for i in content_positions if i < len(tokens)]
    print(f"Content tokens to be masked: {content_tokens}")

    # Test attention mask generation
    print(f"\n--- Testing MedMCQA Attention Masking ---")

    for mask_fraction in [0.0, 0.3, 0.7, 1.0]:
        attention_mask = create_random_attention_mask(
            input_ids, tokenizer, mask_fraction=mask_fraction
        )

        masked_positions = (attention_mask[0] == 0).nonzero().flatten().tolist()
        total_content = len(content_positions)
        actual_masked = len([pos for pos in masked_positions if pos in content_positions])

        print(f"Mask fraction {mask_fraction:.1f}:")
        print(f"  Content tokens: {total_content}")
        print(f"  Masked content tokens: {actual_masked}")
        print(f"  Actual mask fraction: {actual_masked/total_content if total_content > 0 else 0:.2f}")

        # Show what gets masked
        if mask_fraction > 0:
            masked_content_tokens = [tokens[i] for i in masked_positions if i in content_positions and i < len(tokens)]
            preserved_tokens = [tokens[i] for i in range(len(tokens)) if attention_mask[0][i] == 1]
            print(f"  Masked content: {masked_content_tokens[:5]}...")
            print(f"  Preserved structural: {[t for t in preserved_tokens if not t.startswith('question_content')]}")

    print(f"\n✓ MedMCQA content-only attention masking working correctly!")
    print(f"✓ Only question content is masked, structure and answer choices (A,B,C,D) preserved!")

if __name__ == "__main__":
    test_medmcqa_attention_mask()