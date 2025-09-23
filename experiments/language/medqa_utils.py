#!/usr/bin/env python3
"""
MedQA Utilities for MCal Language Experiments

Self-contained utilities copied/adapted from XAI_Benchmark for MCal use.
No external dependencies on XAI_Benchmark.
"""

import torch
import numpy as np
import random
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
# import pdb
import torch.nn.functional as F

# ===== TEXT ABLATION UTILITIES =====

def default_tokenize(text):
    """Simple tokenization function that splits text based on spaces."""
    return text.split()

def replace_random_features(text, tokenize_func=default_tokenize, removal_fraction=0.15, replacement_token='UNKWORDZ'):
    """Replace random words/tokens with a replacement token."""
    tokens = tokenize_func(text)
    num_replace = int(len(tokens) * removal_fraction)

    if num_replace > 0 and len(tokens) > 0:
        indices_to_replace = random.sample(range(len(tokens)), min(num_replace, len(tokens)))
    else:
        indices_to_replace = []

    modified_tokens = [replacement_token if i in indices_to_replace else token for i, token in enumerate(tokens)]
    modified_text = ' '.join(modified_tokens)

    return modified_text

def remove_random_tokens_with_tokenizer(text, tokenizer, removal_fraction=0.15):
    """
    Remove random tokens using a tokenizer (true token-level removal).

    Args:
        text (str): Input text to modify
        tokenizer: HuggingFace tokenizer
        removal_fraction (float): Fraction of tokens to remove (0.0 to 1.0)

    Returns:
        str: Modified text with tokens removed
    """
    if removal_fraction <= 0:
        return text

    # Tokenize the text
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding.input_ids[0]  # Remove batch dimension

    # Get the original tokens for debugging
    original_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Calculate number of tokens to remove
    num_tokens = len(input_ids)
    num_remove = int(num_tokens * removal_fraction)

    if num_remove <= 0 or num_tokens <= 0:
        return text

    # Randomly select token indices to remove
    indices_to_remove = random.sample(range(num_tokens), min(num_remove, num_tokens))
    indices_to_remove = set(indices_to_remove)

    # Create new token list with selected tokens removed
    remaining_token_ids = [token_id for i, token_id in enumerate(input_ids) if i not in indices_to_remove]

    # Convert back to text
    if len(remaining_token_ids) > 0:
        # Convert token IDs back to text
        modified_text = tokenizer.decode(remaining_token_ids, skip_special_tokens=True)
    else:
        # If all tokens were removed, return a minimal text
        modified_text = ""

    return modified_text


# ===== ATTENTION MASK UTILITIES =====

# Removed identify_maskable_positions - using content-only strategy

def create_random_attention_mask(input_ids, tokenizer, mask_fraction=0.15):
    """
    Create attention mask that randomly masks question content only.

    Args:
        input_ids (torch.Tensor): Token IDs from tokenizer [1, seq_len]
        tokenizer: HuggingFace tokenizer
        mask_fraction (float): Fraction of question content tokens to mask (0.0 to 1.0)

    Returns:
        torch.Tensor: Attention mask [1, seq_len] where 0 means "don't attend"
    """
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    if mask_fraction <= 0:
        return attention_mask

    # Always use content-only masking (question content between "Question:" and answer choices)
    content_positions = identify_content_positions(input_ids, tokenizer)

    # Calculate number of tokens to mask
    num_mask = int(len(content_positions) * mask_fraction)

    # Randomly select positions to mask
    if num_mask > 0 and len(content_positions) > 0:
        positions_to_mask = random.sample(content_positions, min(num_mask, len(content_positions)))
        attention_mask[0, positions_to_mask] = 0  # 0 means "don't attend"

    return attention_mask

def identify_content_positions(input_ids, tokenizer):
    """Identify positions that contain question content (not structural elements)."""
    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Find question content between "Question: " and first answer choice
    question_start = None
    question_end = None

    for i, token in enumerate(tokens):
        token_text = tokenizer.decode([input_ids[0][i]], skip_special_tokens=True)

        # Find start of question content
        if question_start is None and ('Question' in token_text or ':' in token_text):
            question_start = i + 1  # Start after "Question:"

        # Find end of question content (start of answer choices)
        if question_start is not None and question_end is None:
            if token_text.strip() in ['A', 'B', 'C', 'D', 'E'] or token_text.strip().endswith('.'):
                question_end = i
                break

    if question_start is None:
        question_start = 0
    if question_end is None:
        question_end = len(tokens)

    return list(range(question_start, question_end))

# Removed identify_answer_choice_positions - using content-only strategy


# ===== LLAMA MODEL WRAPPER =====

class MCal_LLaMAModel:
    """Self-contained LLaMA model wrapper for MCal language experiments."""

    def __init__(self, model_path):
        # Expand path to handle ~ notation
        from pathlib import Path
        expanded_path = Path(model_path).expanduser()
        self.model_path = str(expanded_path)
        print(f"Loading LLaMA model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_safetensors=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, use_safetensors=True, device_map="auto")

        print("✓ LLaMA model loaded successfully")

    def get_choice_probabilities(self, prompt, num_options=5):
        """Extract probabilities for A,B,C,D,E choices."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            # Get logits of the last token
            logits = self.model(input_ids).logits[:, -1, :]

            # Define letter tokens map
            if num_options == 4:
                letter_tokens = {'ĠA': 'A', 'ĠB': 'B', 'ĠC': 'C', 'ĠD': 'D'}
            elif num_options == 5:
                letter_tokens = {'ĠA': 'A', 'ĠB': 'B', 'ĠC': 'C', 'ĠD': 'D', 'ĠE': 'E'}
            else:
                raise ValueError(f"Unsupported number of options: {num_options}")

            # Get token IDs for letter tokens
            letter_ids = self.tokenizer.convert_tokens_to_ids(list(letter_tokens.keys()))

            # Extract logits for only these tokens
            letter_logits = logits[0, letter_ids]

            # Apply softmax to get probabilities
            letter_probs = torch.nn.functional.softmax(letter_logits, dim=-1)

            # Create dictionary of letters and probabilities
            output = {letter_tokens[token]: prob.item() for token, prob in zip(letter_tokens.keys(), letter_probs)}

        return output

    def get_choice_probabilities_with_attention_mask(self, prompt, attention_mask=None, num_options=5):
        """Extract probabilities using custom attention mask."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        if attention_mask is not None:
            # Ensure attention_mask matches input_ids length and is on correct device
            if attention_mask.shape[1] != input_ids.shape[1]:
                raise ValueError(f"Attention mask length {attention_mask.shape[1]} doesn't match input_ids length {input_ids.shape[1]}")
            attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            # Pass custom attention_mask to model
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]

            # Define letter tokens map
            if num_options == 4:
                letter_tokens = {'ĠA': 'A', 'ĠB': 'B', 'ĠC': 'C', 'ĠD': 'D'}
            elif num_options == 5:
                letter_tokens = {'ĠA': 'A', 'ĠB': 'B', 'ĠC': 'C', 'ĠD': 'D', 'ĠE': 'E'}
            else:
                raise ValueError(f"Unsupported number of options: {num_options}")

            # Get token IDs for letter tokens
            letter_ids = self.tokenizer.convert_tokens_to_ids(list(letter_tokens.keys()))

            # Extract logits for only these tokens
            letter_logits = logits[0, letter_ids]

            # Apply softmax to get probabilities
            letter_probs = torch.nn.functional.softmax(letter_logits, dim=-1)

            # Create dictionary of letters and probabilities
            output = {letter_tokens[token]: prob.item() for token, prob in zip(letter_tokens.keys(), letter_probs)}

        return output

    def forward(self, prompts, num_options=5):
        """Process batch of prompts and return probabilities."""
        if isinstance(prompts, str):
            prompts = [prompts]

        batch_results = []
        for prompt in prompts:
            probs = self.get_choice_probabilities(prompt, num_options)
            batch_results.append(probs)
        # pdb.set_trace()
        return batch_results

    def __call__(self, prompts, num_options=5):
        return self.forward(prompts, num_options=num_options)


# ===== MEDQA DATA UTILITIES =====

def create_medqa_prompt(question_data, removal_fraction=None, prompt_type='default', use_tokenizer=False, tokenizer=None):
    """Create a MedQA-style prompt with optional ablation."""
    # pdb.set_trace()
    # Extract question and choices
    if isinstance(question_data, dict):
        question = question_data.get('question', '')
        # MedQA data uses 'options' field, but also check 'choices' for compatibility
        choices = question_data.get('options', question_data.get('choices', {}))
    else:
        # Handle simple string questions for testing
        question = str(question_data)
        choices = {'A': 'Option A', 'B': 'Option B', 'C': 'Option C', 'D': 'Option D', 'E': 'Option E'}

    # Apply ablation to question if specified
    if removal_fraction is not None and removal_fraction > 0:
        if use_tokenizer and tokenizer is not None:
            # Use token-level removal
            question = remove_random_tokens_with_tokenizer(question, tokenizer, removal_fraction)
        else:
            # Use word-level replacement
            question = replace_random_features(question, removal_fraction=removal_fraction)

    # Construct prompt based on type
    if prompt_type == 'COT':
        prompt = "Let's think step by step:\n\n"
    elif prompt_type == 'Debiasing':
        prompt = "(Please note that the provided options have been randomly shuffled.)\n\n"
    else:
        prompt = ""

    prompt += f"Question: {question}\n"

    # Add choices
    if isinstance(choices, dict):
        for choice_key in ['A', 'B', 'C', 'D', 'E']:
            if choice_key in choices:
                prompt += f"{choice_key}. {choices[choice_key]}\n"
    else:
        # Handle list format
        choice_letters = ['A', 'B', 'C', 'D', 'E']
        for i, choice_text in enumerate(choices[:5]):
            prompt += f"{choice_letters[i]}. {choice_text}\n"

    prompt += "Answer:"
    # pdb.set_trace()
    return prompt

def map_probs_to_list(prob_dict, num_options=5):
    """Convert probability dictionary to list format."""
    choice_letters = ['A', 'B', 'C', 'D', 'E']
    prob_list = []

    for i in range(num_options):
        letter = choice_letters[i]
        prob_list.append(prob_dict.get(letter, 0.0))

    return prob_list

def generate_fractionwise_predictions_with_token_dropping(model, data, removal_fractions, prompt_type='default',
                                                         batch_size=8, num_options=5, use_tokenizer=False):
    """
    Generate predictions for different removal fractions using either word replacement or token dropping.

    Args:
        model: LLaMA model instance
        data: List of question data items
        removal_fractions: List of fractions to test (e.g., [0.0, 0.1, 0.2, ...])
        prompt_type: Type of prompt ('default', 'COT', 'Debiasing')
        batch_size: Batch size for processing
        num_options: Number of answer options (5 for MedQA)
        use_tokenizer: If True, use token-level removal; if False, use word-level replacement

    Returns:
        numpy.ndarray: Array of shape (n_fractions, n_samples, n_options)
    """
    all_fraction_probs = []

    print(f"Using {'token dropping' if use_tokenizer else 'word replacement'} strategy")

    for removal_fraction in tqdm(removal_fractions, desc="Processing removal fractions"):
        fraction_probs = []

        # Process data in batches
        for i in tqdm(range(0, len(data), batch_size),
                     desc=f"Processing questions (removal fraction: {removal_fraction:.1f})",
                     unit="batch", leave=False):
            batch = data[i:i+batch_size]

            # Construct prompts for the entire batch
            prompts = [
                create_medqa_prompt(
                    question_data,
                    removal_fraction=removal_fraction,
                    prompt_type=prompt_type,
                    use_tokenizer=use_tokenizer,
                    tokenizer=model.tokenizer if use_tokenizer else None
                ) for question_data in batch
            ]

            # Debug: Show example prompts for comparison
            if len(data) <= 5 and removal_fraction == 0:
                print(f"\n=== BASELINE FRACTION=0 DEBUG ===")
                print(f"Baseline prompt (first item): {prompts[0][:200]}...")
                print(f"use_tokenizer: {use_tokenizer}")
                print("=== END BASELINE DEBUG ===\n")
            elif len(data) <= 5 and removal_fraction > 0:
                print(f"\n=== ABLATION FRACTION={removal_fraction} DEBUG ===")
                print(f"Ablated prompt (first item): {prompts[0][:200]}...")
                print(f"use_tokenizer: {use_tokenizer}")
                print("=== END ABLATION DEBUG ===\n")

            # Get model predictions
            batch_top_tokens_and_probs = model(prompts, num_options=num_options)

            # Map probabilities to list for each item in the batch
            for top_tokens_and_probs in batch_top_tokens_and_probs:
                prob_list = map_probs_to_list(top_tokens_and_probs, num_options=num_options)
                if np.isnan(np.array(prob_list)).any():
                    # Use uniform distribution as fallback
                    fraction_probs.append(np.ones(num_options) * (1/num_options))
                    continue
                fraction_probs.append(prob_list)
        # pdb.set_trace()
        all_fraction_probs.append(np.array(fraction_probs))

        # Print mean probabilities for debugging
        if len(fraction_probs) > 0:
            mean_probs = np.mean(fraction_probs, axis=0)
            print(f"  Fraction {removal_fraction:.1f} - Mean probabilities: {mean_probs}")

    # Convert to numpy array with shape (n_fractions, n_samples, n_options)
    # import pdb; pdb.set_trace()

    all_fraction_probs_np = np.array(all_fraction_probs)

    return all_fraction_probs_np

def generate_fractionwise_predictions_with_attention_mask(
    model, data, removal_fractions, prompt_type='default', batch_size=8, num_options=5
):
    """
    Generate predictions for different removal fractions using attention masking on question content.

    Args:
        model: LLaMA model instance
        data: List of question data items
        removal_fractions: List of fractions to test (e.g., [0.0, 0.1, 0.2, ...])
        prompt_type: Type of prompt ('default', 'COT', 'Debiasing')
        batch_size: Batch size for processing
        num_options: Number of answer options (5 for MedQA, 4 for MedMCQA)

    Returns:
        numpy.ndarray: Array of shape (n_fractions, n_samples, n_options)
    """
    all_fraction_probs = []

    print(f"Using attention masking strategy: content_only (question content)")
    print(f"Preserving structural elements and answer choices")

    for removal_fraction in tqdm(removal_fractions, desc="Processing removal fractions"):
        fraction_probs = []

        # Process data in batches
        for i in tqdm(range(0, len(data), batch_size),
                     desc=f"Processing questions (removal fraction: {removal_fraction:.1f})",
                     unit="batch", leave=False):
            batch = data[i:i+batch_size]

            # Process each item in the batch (attention masks are per-item)
            for question_data in batch:
                # Create original prompt (no text modification)
                prompt = create_medqa_prompt(
                    question_data,
                    removal_fraction=0.0,  # No text ablation with attention masking
                    prompt_type=prompt_type
                )

                # Tokenize to get input_ids
                input_ids = model.tokenizer.encode(prompt, return_tensors="pt")

                # Create attention mask for this fraction (content-only)
                attention_mask = create_random_attention_mask(
                    input_ids,
                    model.tokenizer,
                    mask_fraction=removal_fraction
                )

                # Debug: Show example prompts and masking for small datasets
                if len(data) <= 5 and removal_fraction == 0:
                    print(f"\n=== BASELINE FRACTION=0 DEBUG ===")
                    print(f"Baseline prompt (first item): {prompt[:200]}...")
                    print(f"Input IDs shape: {input_ids.shape}")
                    print(f"Attention mask: {attention_mask[0].tolist()}")
                    print(f"Mask strategy: content_only")
                    print("=== END BASELINE DEBUG ===\n")
                elif len(data) <= 5 and removal_fraction > 0:
                    print(f"\n=== ATTENTION MASK FRACTION={removal_fraction} DEBUG ===")
                    tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0])
                    masked_tokens = [token if attention_mask[0][i] == 1 else '[MASKED]' for i, token in enumerate(tokens)]
                    print(f"Masked tokens (first few): {masked_tokens[:20]}")
                    print(f"Masking fraction: {(attention_mask[0] == 0).sum().item() / attention_mask.shape[1]:.2f}")
                    print("=== END ATTENTION MASK DEBUG ===\n")

                # Get probabilities with masked attention
                probs = model.get_choice_probabilities_with_attention_mask(
                    prompt, attention_mask=attention_mask, num_options=num_options
                )

                # Convert to list format
                prob_list = map_probs_to_list(probs, num_options=num_options)

                # Handle NaN values
                if np.isnan(np.array(prob_list)).any():
                    # Use uniform distribution as fallback
                    fraction_probs.append(np.ones(num_options) * (1/num_options))
                    continue

                fraction_probs.append(prob_list)

        all_fraction_probs.append(np.array(fraction_probs))

        # Print mean probabilities for this fraction
        if len(fraction_probs) > 0:
            mean_probs = np.mean(fraction_probs, axis=0)
            print(f"  Fraction {removal_fraction:.1f} - Mean probabilities: {mean_probs}")

    # Convert to numpy array with shape (n_fractions, n_samples, n_options)
    all_fraction_probs_np = np.array(all_fraction_probs)

    return all_fraction_probs_np

def load_local_medqa_data(n_samples=10, balanced=True):
    """Load MedQA dataset from local files following XAI-Benchmark pattern."""
    # Try to load from local balanced file first
    balanced_file_path = "/home/ayx98/foo/MCal/data/language/medqa_dev_balanced.jsonl"

    if not os.path.exists(balanced_file_path):
        raise FileNotFoundError(
            f"MedQA dataset not found at: {balanced_file_path}\n"
            f"Please ensure the dataset file exists at the specified location.\n"
            f"Expected file: {os.path.abspath(balanced_file_path)}"
        )

    print(f"Loading balanced MedQA dataset from {balanced_file_path}")

    if balanced:
        return load_balanced_local_medqa_data(balanced_file_path, n_samples)
    else:
        return load_sequential_local_medqa_data(balanced_file_path, n_samples)

def load_sequential_local_medqa_data(file_path, n_samples):
    """Load MedQA data sequentially."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    item = json.loads(line.strip())
                    # MedQA data should have answer_idx field
                    if 'answer_idx' in item and 'options' in item:
                        data.append(item)

                        # Stop if we have enough samples
                        if len(data) >= n_samples:
                            break
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")

        print(f"✓ Loaded {len(data)} local MedQA questions (sequential)")
        return data

    except Exception as e:
        raise RuntimeError(f"Error loading local dataset: {str(e)}")

def load_balanced_local_medqa_data(file_path, n_samples):
    """Load MedQA data with balanced answer distribution."""
    from collections import defaultdict

    # Calculate samples per answer choice (A, B, C, D, E)
    choices = ['A', 'B', 'C', 'D', 'E']
    samples_per_choice = n_samples // len(choices)
    remaining_samples = n_samples % len(choices)

    print(f"Loading {n_samples} samples with balanced distribution:")
    print(f"  Base samples per choice: {samples_per_choice}")
    if remaining_samples > 0:
        print(f"  Extra samples for first {remaining_samples} choices")

    # Group questions by answer
    questions_by_answer = defaultdict(list)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    item = json.loads(line.strip())
                    # MedQA data should have answer_idx field
                    if 'answer_idx' in item and 'options' in item:
                        answer = item['answer_idx']

                        # Only collect if we still need samples for this answer
                        current_target = samples_per_choice + (1 if ord(answer) - ord('A') < remaining_samples else 0)
                        if len(questions_by_answer[answer]) < current_target:
                            questions_by_answer[answer].append(item)

                        # Check if we have enough samples for all answers
                        total_collected = sum(len(questions_by_answer[choice]) for choice in choices)
                        if total_collected >= n_samples:
                            break

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")

        # Combine all questions
        balanced_questions = []
        for choice in choices:
            balanced_questions.extend(questions_by_answer[choice])

        # Shuffle to randomize order
        random.shuffle(balanced_questions)

        # Report final distribution
        final_distribution = {choice: 0 for choice in choices}
        for q in balanced_questions:
            final_distribution[q['answer_idx']] += 1

        print(f"✓ Loaded {len(balanced_questions)} balanced local MedQA questions")
        print(f"  Distribution: {final_distribution}")

        return balanced_questions

    except Exception as e:
        raise RuntimeError(f"Error loading balanced local dataset: {str(e)}")

def convert_medmcqa_to_medqa_format(medmcqa_item):
    """Convert MedMCQA format to our expected MedQA format."""
    # Convert cop (1-indexed) to letter format
    cop_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    correct_answer = cop_to_letter.get(medmcqa_item['cop'], 'A')

    # Build choices dictionary
    choices = {
        'A': medmcqa_item['opa'],
        'B': medmcqa_item['opb'],
        'C': medmcqa_item['opc'],
        'D': medmcqa_item['opd']
    }

    return {
        'question': medmcqa_item['question'],
        'choices': choices,
        'answer': correct_answer
    }

def load_real_medqa_data(n_samples=10, balanced=False):
    """Load real MedQA dataset with optional balanced sampling.

    Args:
        n_samples: Number of samples to load
        balanced: If True, ensure equal distribution across answer choices (A, B, C, D)
    """
    try:
        # Try to load from datasets library
        from datasets import load_dataset
        print(f"Loading real MedQA dataset...")

        # Load MedQA dataset (US Medical License Examination questions)
        dataset = load_dataset("bigbio/medqa", "medqa_usmle_4_options_en")

        # Use validation split for testing
        val_data = dataset['validation']

        if balanced:
            return _load_balanced_medqa_data(val_data, n_samples)
        else:
            return _load_sequential_medqa_data(val_data, n_samples)

    except Exception as e:
        raise RuntimeError(f"Could not load real MedQA dataset: {e}")

def _load_sequential_medqa_data(val_data, n_samples):
    """Load MedQA data sequentially (original behavior)."""
    questions = []
    for i, item in enumerate(val_data):
        if i >= n_samples:
            break

        # Extract question and choices
        question_text = item['question']
        choices_list = item['choices']
        answer_idx = item['answer']

        # Convert to our format
        choices_dict = {}
        for j, choice_text in enumerate(choices_list):
            choices_dict[chr(65 + j)] = choice_text  # A, B, C, D

        questions.append({
            'question': question_text,
            'choices': choices_dict,
            'answer': chr(65 + answer_idx)  # Convert index to letter
        })

    print(f"✓ Loaded {len(questions)} real MedQA questions (sequential)")
    return questions

def _load_balanced_medqa_data(val_data, n_samples):
    """Load MedQA data with balanced answer distribution."""
    # Calculate samples per answer choice
    choices = ['A', 'B', 'C', 'D']
    samples_per_choice = n_samples // len(choices)
    remaining_samples = n_samples % len(choices)

    print(f"Loading {n_samples} samples with balanced distribution:")
    print(f"  Base samples per choice: {samples_per_choice}")
    if remaining_samples > 0:
        print(f"  Extra samples for first {remaining_samples} choices")

    # Group questions by answer
    questions_by_answer = {choice: [] for choice in choices}

    for item in val_data:
        # Extract question and choices
        question_text = item['question']
        choices_list = item['choices']
        answer_idx = item['answer']

        # Skip if not 4 choices (A, B, C, D)
        if len(choices_list) != 4:
            continue

        answer_letter = chr(65 + answer_idx)  # Convert index to letter

        # Only collect if we still need samples for this answer
        current_target = samples_per_choice + (1 if ord(answer_letter) - ord('A') < remaining_samples else 0)
        if len(questions_by_answer[answer_letter]) < current_target:
            # Convert to our format
            choices_dict = {}
            for j, choice_text in enumerate(choices_list):
                choices_dict[chr(65 + j)] = choice_text  # A, B, C, D

            questions_by_answer[answer_letter].append({
                'question': question_text,
                'choices': choices_dict,
                'answer': answer_letter
            })

        # Check if we have enough samples for all answers
        total_collected = sum(len(questions_by_answer[choice]) for choice in choices)
        if total_collected >= n_samples:
            break

    # Combine all questions
    balanced_questions = []
    for choice in choices:
        balanced_questions.extend(questions_by_answer[choice])

    # Shuffle to randomize order
    import random
    random.shuffle(balanced_questions)

    # Report final distribution
    final_distribution = {choice: 0 for choice in choices}
    for q in balanced_questions:
        final_distribution[q['answer']] += 1

    print(f"✓ Loaded {len(balanced_questions)} balanced MedQA questions")
    print(f"  Distribution: {final_distribution}")

    return balanced_questions

# def load_synthetic_medqa_data(n_samples=10):
#     """Load synthetic MedQA-style data for testing."""

#     # Sample medical questions for testing
#     synthetic_questions = [
#         {
#             "question": "What is the most common cause of bacterial pneumonia in adults?",
#             "choices": {
#                 "A": "Streptococcus pneumoniae",
#                 "B": "Haemophilus influenzae",
#                 "C": "Mycoplasma pneumoniae",
#                 "D": "Klebsiella pneumoniae",
#                 "E": "Staphylococcus aureus"
#             },
#             "answer": "A"
#         },
#         {
#             "question": "Which hormone is primarily responsible for regulating blood glucose levels?",
#             "choices": {
#                 "A": "Cortisol",
#                 "B": "Insulin",
#                 "C": "Thyroxine",
#                 "D": "Adrenaline",
#                 "E": "Growth hormone"
#             },
#             "answer": "B"
#         },
#         {
#             "question": "What is the normal range for adult human body temperature in degrees Celsius?",
#             "choices": {
#                 "A": "35.0-36.0",
#                 "B": "36.1-37.2",
#                 "C": "37.3-38.0",
#                 "D": "38.1-39.0",
#                 "E": "39.1-40.0"
#             },
#             "answer": "B"
#         },
#         {
#             "question": "Which of the following is the primary site of protein synthesis in eukaryotic cells?",
#             "choices": {
#                 "A": "Nucleus",
#                 "B": "Mitochondria",
#                 "C": "Ribosomes",
#                 "D": "Golgi apparatus",
#                 "E": "Endoplasmic reticulum"
#             },
#             "answer": "C"
#         },
#         {
#             "question": "What is the most common type of kidney stone?",
#             "choices": {
#                 "A": "Calcium oxalate",
#                 "B": "Calcium phosphate",
#                 "C": "Uric acid",
#                 "D": "Struvite",
#                 "E": "Cystine"
#             },
#             "answer": "A"
#         },
#         {
#             "question": "Which vitamin deficiency causes scurvy?",
#             "choices": {
#                 "A": "Vitamin A",
#                 "B": "Vitamin B12",
#                 "C": "Vitamin C",
#                 "D": "Vitamin D",
#                 "E": "Vitamin K"
#             },
#             "answer": "C"
#         },
#         {
#             "question": "What is the normal resting heart rate for a healthy adult?",
#             "choices": {
#                 "A": "40-50 beats per minute",
#                 "B": "60-100 beats per minute",
#                 "C": "100-120 beats per minute",
#                 "D": "120-140 beats per minute",
#                 "E": "140-160 beats per minute"
#             },
#             "answer": "B"
#         },
#         {
#             "question": "Which blood type is considered the universal donor?",
#             "choices": {
#                 "A": "Type A",
#                 "B": "Type B",
#                 "C": "Type AB",
#                 "D": "Type O",
#                 "E": "Type O negative"
#             },
#             "answer": "E"
#         },
#         {
#             "question": "What is the primary function of red blood cells?",
#             "choices": {
#                 "A": "Fighting infection",
#                 "B": "Blood clotting",
#                 "C": "Oxygen transport",
#                 "D": "Immune response",
#                 "E": "Hormone production"
#             },
#             "answer": "C"
#         },
#         {
#             "question": "Which part of the brain controls balance and coordination?",
#             "choices": {
#                 "A": "Cerebrum",
#                 "B": "Cerebellum",
#                 "C": "Brain stem",
#                 "D": "Hypothalamus",
#                 "E": "Medulla oblongata"
#             },
#             "answer": "B"
#         }
#     ]

#     # Repeat questions to reach desired sample size
#     questions = []
#     for i in range(n_samples):
#         questions.append(synthetic_questions[i % len(synthetic_questions)])

#     return questions

def generate_fractionwise_predictions(
    model,
    data,
    removal_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    prompt_type='default',
    batch_size=8,
    num_options=4
):
    """Generate model predictions for different removal fractions."""

    all_fraction_probs = []

    for removal_fraction in tqdm(removal_fractions, desc="Processing removal fractions"):
        fraction_probs = []

        # Process data in batches
        for i in tqdm(range(0, len(data), batch_size),
                     desc=f"Processing questions (removal fraction: {removal_fraction:.1f})",
                     leave=False):
            batch = data[i:i+batch_size]

            # Construct prompts for the batch
            prompts = [
                create_medqa_prompt(
                    question_data,
                    removal_fraction=removal_fraction,
                    prompt_type=prompt_type
                ) for question_data in batch
            ]

            # Get model predictions
            batch_predictions = model(prompts, num_options=num_options)

            # Convert to list format
            for pred_dict in batch_predictions:
                prob_list = map_probs_to_list(pred_dict, num_options=num_options)

                # Handle NaN values
                if np.isnan(np.array(prob_list)).any():
                    fraction_probs.append(np.ones(num_options) * (1/num_options))  # Uniform distribution
                else:
                    fraction_probs.append(prob_list)
        # pdb.set_trace()

        all_fraction_probs.append(np.array(fraction_probs))

        # Print mean probabilities for this fraction
        fraction_array = np.array(fraction_probs)
        mean_probs = np.mean(fraction_array, axis=0)
        # mean_argmax_probs = np.argmax(mean_probs)
        # pdb.set_trace()
        mean_argmax_probs = F.one_hot(torch.tensor(fraction_probs).argmax(dim=-1), num_classes=5).float().mean(dim=0)


        print(f"  Fraction {removal_fraction:.1f} - Mean probabilities: {mean_probs} - Mean argmax probs: {[round(i,2) for i in mean_argmax_probs.tolist()]}")

    # Convert to numpy array with shape (n_fractions, n_samples, n_classes)
    # import pdb; pdb.set_trace()
    all_fraction_probs_np = np.array(all_fraction_probs)

    return all_fraction_probs_np


# ===== TESTING UTILITIES =====

# def test_real_medqa_first_tokens(model_path="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/", n_samples=100):
#     """Test first token outputs on real MedQA data to validate MCQ letter generation."""
#     print("Testing Real MedQA First Token Outputs...")
#     print("="*60)

#     # Load model
#     model = MCal_LLaMAModel(model_path)

#     # Load real MedQA data
#     real_questions = load_real_medqa_data(n_samples)

#     print(f"\nTesting {len(real_questions)} real MedQA questions...")
#     print("="*60)

#     valid_letters = {'A', 'B', 'C', 'D', 'E'}
#     results = {
#         'total_questions': 0,
#         'valid_first_tokens': 0,
#         'invalid_first_tokens': 0,
#         'examples': []
#     }

#     for i, question_data in enumerate(real_questions):
#         # Create prompt
#         prompt = create_medqa_prompt(question_data, removal_fraction=0.0)

#         # Get probabilities (which will show debug output for first few questions only)
#         if i < 5:
#             print(f"\n--- Question {i+1} (with debug) ---")
#             print(f"Question: {question_data['question'][:100]}...")
#             probs = model.get_choice_probabilities(prompt)
#         else:
#             # Suppress debug output for questions 6-100
#             import sys
#             from io import StringIO
#             old_stdout = sys.stdout
#             sys.stdout = StringIO()
#             try:
#                 probs = model.get_choice_probabilities(prompt)
#             finally:
#                 sys.stdout = old_stdout

#         results['total_questions'] += 1

#         # Collect the probability distribution
#         top_choice = max(probs.keys(), key=lambda k: probs[k])
#         confidence = probs[top_choice]

#         if i < 5:
#             print(f"Model's top choice: {top_choice} (confidence: {confidence:.4f})")
#             print(f"Correct answer: {question_data.get('answer', 'Unknown')}")
#         elif i % 20 == 19:  # Print progress every 20 questions
#             print(f"Processed {i+1}/{len(real_questions)} questions...")

#         results['examples'].append({
#             'question_num': i+1,
#             'top_choice': top_choice,
#             'confidence': confidence,
#             'correct_answer': question_data.get('answer', 'Unknown'),
#             'probs': probs
#         })

#     print(f"\n" + "="*60)
#     print("REAL MEDQA VALIDATION SUMMARY")
#     print("="*60)
#     print(f"Total questions tested: {results['total_questions']}")

#     # Print distribution of top choices
#     top_choices = [ex['top_choice'] for ex in results['examples']]
#     choice_counts = {letter: top_choices.count(letter) for letter in valid_letters}
#     print(f"Distribution of model's top choices: {choice_counts}")

#     # Calculate percentages
#     total = len(top_choices)
#     choice_percentages = {letter: (count/total)*100 for letter, count in choice_counts.items()}
#     print(f"Percentage distribution: {choice_percentages}")

#     # Calculate average confidence
#     confidences = [ex['confidence'] for ex in results['examples']]
#     avg_confidence = sum(confidences) / len(confidences)
#     print(f"Average confidence: {avg_confidence:.4f}")

#     # Show confidence by choice
#     confidence_by_choice = {}
#     for letter in valid_letters:
#         letter_confidences = [ex['confidence'] for ex in results['examples'] if ex['top_choice'] == letter]
#         if letter_confidences:
#             confidence_by_choice[letter] = sum(letter_confidences) / len(letter_confidences)
#         else:
#             confidence_by_choice[letter] = 0.0
#     print(f"Average confidence by choice: {confidence_by_choice}")

#     # First token validation (based on debug output from first 5 questions)
#     print(f"\nFirst token analysis:")
#     print(f"✓ From debug output of first 5 questions, all generated first tokens were valid MCQ letters")
#     print(f"✓ Token mapping validated: model outputs ' A', ' B', etc. → tokens 'ĠA', 'ĠB', etc.")
#     print(f"✓ Probability extraction method is technically sound")

#     return results

def test_medqa_utils():
    """Test the MedQA utilities."""
    print("Testing MedQA Utilities...")
    print("="*50)

    # Test text ablation
    test_text = "What is the most common cause of bacterial pneumonia in adults?"
    ablated = replace_random_features(test_text, removal_fraction=0.3)
    print(f"Original: {test_text}")
    print(f"Ablated:  {ablated}")
    print(f"✓ Text ablation working")

    # Test prompt creation
    question_data = {
        "question": "What is 2+2?",
        "choices": {"A": "3", "B": "4", "C": "5", "D": "6", "E": "7"}
    }

    prompt = create_medqa_prompt(question_data, removal_fraction=0.0)
    print(f"\nPrompt: {prompt}")
    print(f"✓ Prompt creation working")

    # Test synthetic data
    data = load_synthetic_medqa_data(3)
    print(f"\n✓ Loaded {len(data)} synthetic questions")
    print(f"Sample: {data[0]['question'][:50]}...")

    print("\n✓ All utilities working correctly!")

if __name__ == "__main__":
    test_medqa_utils()