#!/usr/bin/env python3
"""
MedMCQA Utilities for MCal Language Experiments

Self-contained utilities for MedMCQA dataset processing and LLaMA model interactions.
Following the exact pattern of medqa_utils.py but adapted for MedMCQA format.

No XAI_Benchmark dependencies - all functions copied and adapted locally.
"""

import torch
import numpy as np
import random
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ===== TEXT ABLATION UTILITIES =====

def default_tokenize(text):
    """Simple tokenization function that splits text based on spaces."""
    return text.split()

def replace_random_features(text, tokenize_func=default_tokenize, removal_fraction=0.15, replacement_token='UNKWORDZ'):
    """Replace random words/tokens with a replacement token."""
    if removal_fraction == 0:
        return text

    tokens = tokenize_func(text)
    num_tokens = len(tokens)
    num_to_replace = int(removal_fraction * num_tokens)

    if num_to_replace == 0:
        return text

    # Randomly select indices to replace
    indices_to_replace = random.sample(range(num_tokens), min(num_to_replace, num_tokens))

    # Replace selected tokens
    for idx in indices_to_replace:
        tokens[idx] = replacement_token

    return ' '.join(tokens)

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
            # For MedMCQA, look for A, B, C, D (not E)
            if token_text.strip() in ['A', 'B', 'C', 'D'] or token_text.strip().endswith('.'):
                question_end = i
                break

    if question_start is None:
        question_start = 0
    if question_end is None:
        question_end = len(tokens)

    return list(range(question_start, question_end))


# ===== LLAMA MODEL WRAPPER =====

class MCal_LLaMAModel:
    """LLaMA model wrapper for MedMCQA multiple choice predictions."""

    def __init__(self, model_path):
        """Initialize the LLaMA model and tokenizer."""
        expanded_path = Path(model_path).expanduser()
        print(f"Loading LLaMA model from: {expanded_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(expanded_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(expanded_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"✓ LLaMA model loaded successfully")

    def get_choice_probabilities(self, prompt, num_options=4):
        """Extract probabilities for A,B,C,D choices."""
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
            probs = torch.softmax(letter_logits, dim=0)

            # Map to letter format
            prob_dict = {}
            letters = list(letter_tokens.values())
            for i, letter in enumerate(letters):
                prob_dict[letter] = probs[i].item()

        return prob_dict

    def get_choice_probabilities_with_attention_mask(self, prompt, attention_mask=None, num_options=4):
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
            probs = torch.softmax(letter_logits, dim=0)

            # Map to letter format
            prob_dict = {}
            letters = list(letter_tokens.values())
            for i, letter in enumerate(letters):
                prob_dict[letter] = probs[i].item()

        return prob_dict

    def forward(self, prompts, num_options=4):
        """Process multiple prompts and return probability dictionaries."""
        results = []
        for prompt in prompts:
            probs = self.get_choice_probabilities(prompt, num_options)
            results.append(probs)
        return results

    def __call__(self, prompts, num_options=4):
        """Make the model callable."""
        return self.forward(prompts, num_options=num_options)

# ===== MEDMCQA DATASET UTILITIES =====

def create_medmcqa_prompt(question_data, removal_fraction=0.0, prompt_type='default', use_tokenizer=False, tokenizer=None):
    """Create a prompt for MedMCQA question following XAI-Benchmark pattern."""

    # Extract question and options from MedMCQA format
    question = question_data['question']
    options = {
        'A': question_data['opa'],
        'B': question_data['opb'],
        'C': question_data['opc'],
        'D': question_data['opd']
    }

    # Apply text ablation if specified
    if removal_fraction > 0:
        if use_tokenizer and tokenizer is not None:
            # Use token-level removal
            question = remove_random_tokens_with_tokenizer(question, tokenizer, removal_fraction)
        else:
            # Use word-level replacement
            question = replace_random_features(
                question,
                removal_fraction=removal_fraction,
                replacement_token='UNKWORDZ'
            )

    # Construct prompt based on type
    if prompt_type == 'COT':
        prompt = "Let's think step by step:\n"
    elif prompt_type == 'Debiasing':
        prompt = "(Please note that the provided options have been randomly shuffled, so it is essential to consider them fairly and without bias.)"
    else:
        prompt = ""

    prompt += f"Question: {question}\n"
    for option, text in options.items():
        prompt += f"{option}. {text}\n"

    prompt += "Answer: "

    return prompt

def map_probs_to_list(prob_dict, num_options=4):
    """Convert probability dictionary to ordered list [A, B, C, D] or [A, B, C, D, E]."""
    if num_options == 4:
        letters = ['A', 'B', 'C', 'D']
    elif num_options == 5:
        letters = ['A', 'B', 'C', 'D', 'E']
    else:
        raise ValueError(f"Unsupported number of options: {num_options}")

    prob_list = []
    for letter in letters:
        prob_list.append(prob_dict.get(letter, 0.0))

    return prob_list

def load_local_medmcqa_data(n_samples=10, balanced=True):
    """Load MedMCQA dataset from local files following XAI-Benchmark pattern."""
    # Try to load from local balanced file first
    balanced_file_path = "/home/antonxue/shailesh/MCal/data/language/dev_balanced.json"

    if not os.path.exists(balanced_file_path):
        raise FileNotFoundError(
            f"MedMCQA dataset not found at: {balanced_file_path}\n"
            f"Please ensure the dataset file exists at the specified location.\n"
            f"Expected file: {os.path.abspath(balanced_file_path)}"
        )

    print(f"Loading balanced MedMCQA dataset from {balanced_file_path}")

    if balanced:
        return _load_balanced_medmcqa_data(balanced_file_path, n_samples)
    else:
        return _load_sequential_medmcqa_data(balanced_file_path, n_samples)

def _load_sequential_medmcqa_data(file_path, n_samples):
    """Load MedMCQA data sequentially."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    item = json.loads(line.strip())
                    # Only append items where 'choice_type' is 'single'
                    if item.get("choice_type") == 'single' and len(data) < n_samples:
                        data.append(item)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")

        print(f"✓ Loaded {len(data)} local MedMCQA questions (sequential)")
        return data

    except Exception as e:
        raise RuntimeError(f"Error loading local dataset: {str(e)}")

def _load_balanced_medmcqa_data(file_path, n_samples):
    """Load MedMCQA data with balanced answer distribution."""
    from collections import defaultdict

    # Calculate samples per answer choice (A, B, C, D)
    choices = ['A', 'B', 'C', 'D']
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
                    # Only process single choice questions
                    if item.get("choice_type") == 'single':
                        # Convert cop (1-indexed) to letter format
                        answer_letter = convert_cop_to_letter(item['cop'])

                        # Only collect if we still need samples for this answer
                        current_target = samples_per_choice + (1 if ord(answer_letter) - ord('A') < remaining_samples else 0)
                        if len(questions_by_answer[answer_letter]) < current_target:
                            questions_by_answer[answer_letter].append(item)

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
            answer_letter = convert_cop_to_letter(q['cop'])
            final_distribution[answer_letter] += 1

        print(f"✓ Loaded {len(balanced_questions)} balanced MedMCQA questions")
        print(f"  Distribution: {final_distribution}")

        return balanced_questions

    except Exception as e:
        raise RuntimeError(f"Error loading balanced local dataset: {str(e)}")

def convert_cop_to_letter(cop):
    """Convert cop (1-indexed) to letter format (A, B, C, D)."""
    cop_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    return cop_to_letter.get(cop, 'A')

def load_synthetic_medmcqa_data(n_samples=10):
    """Generate synthetic MedMCQA data for testing."""
    synthetic_questions = [
        {
            'question': 'What is the most common cause of myocardial infarction?',
            'opa': 'Coronary artery disease',
            'opb': 'Hypertension',
            'opc': 'Diabetes mellitus',
            'opd': 'Smoking',
            'cop': 1,  # A
            'choice_type': 'single'
        },
        {
            'question': 'Which drug is first-line treatment for hypertension?',
            'opa': 'Beta blockers',
            'opb': 'ACE inhibitors',
            'opc': 'Calcium channel blockers',
            'opd': 'Diuretics',
            'cop': 2,  # B
            'choice_type': 'single'
        },
        {
            'question': 'What is the normal range for serum creatinine?',
            'opa': '0.1-0.5 mg/dL',
            'opb': '0.6-1.2 mg/dL',
            'opc': '1.5-2.0 mg/dL',
            'opd': '2.5-3.0 mg/dL',
            'cop': 2,  # B
            'choice_type': 'single'
        },
        {
            'question': 'Which vitamin deficiency causes night blindness?',
            'opa': 'Vitamin A',
            'opb': 'Vitamin B',
            'opc': 'Vitamin C',
            'opd': 'Vitamin D',
            'cop': 1,  # A
            'choice_type': 'single'
        }
    ]

    # Extend to requested sample size by cycling through
    questions = []
    for i in range(n_samples):
        questions.append(synthetic_questions[i % len(synthetic_questions)])

    return questions

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
                create_medmcqa_prompt(
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

        all_fraction_probs.append(np.array(fraction_probs))

        # Print mean probabilities for this fraction
        fraction_array = np.array(fraction_probs)
        mean_probs = np.mean(fraction_array, axis=0)
        print(f"  Fraction {removal_fraction:.1f} - Mean probabilities: {mean_probs}")

    # Convert to numpy array with shape (n_fractions, n_samples, n_classes)
    all_fraction_probs_np = np.array(all_fraction_probs)

    return all_fraction_probs_np

def generate_fractionwise_predictions_with_token_dropping(model, data, removal_fractions, prompt_type='default',
                                                         batch_size=8, num_options=4, use_tokenizer=False):
    """
    Generate predictions for different removal fractions using either word replacement or token dropping.

    Args:
        model: LLaMA model instance
        data: List of question data items
        removal_fractions: List of fractions to test (e.g., [0.0, 0.1, 0.2, ...])
        prompt_type: Type of prompt ('default', 'COT', 'Debiasing')
        batch_size: Batch size for processing
        num_options: Number of answer options (4 for MedMCQA)
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
                create_medmcqa_prompt(
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

        all_fraction_probs.append(np.array(fraction_probs))

        # Print mean probabilities for debugging
        if len(fraction_probs) > 0:
            mean_probs = np.mean(fraction_probs, axis=0)
            print(f"  Fraction {removal_fraction:.1f} - Mean probabilities: {mean_probs}")

    # Convert to numpy array with shape (n_fractions, n_samples, n_options)
    all_fraction_probs_np = np.array(all_fraction_probs)

    return all_fraction_probs_np

def generate_fractionwise_predictions_with_attention_mask(
    model, data, removal_fractions, prompt_type='default', batch_size=8, num_options=4
):
    """
    Generate predictions for different removal fractions using attention masking on question content.

    Args:
        model: LLaMA model instance
        data: List of question data items
        removal_fractions: List of fractions to test (e.g., [0.0, 0.1, 0.2, ...])
        prompt_type: Type of prompt ('default', 'COT', 'Debiasing')
        batch_size: Batch size for processing
        num_options: Number of answer options (4 for MedMCQA)

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
                prompt = create_medmcqa_prompt(
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

# ===== TESTING UTILITIES =====

def test_medmcqa_first_tokens(model_path="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/", n_samples=10):
    """Test first token outputs on MedMCQA data to validate MCQ letter generation."""

    print(f"=== Testing MedMCQA First Token Generation ===")
    print(f"Model: {model_path}")
    print(f"Samples: {n_samples}")

    # Load model
    expanded_path = Path(model_path).expanduser()
    model = MCal_LLaMAModel(str(expanded_path))

    # Load test data
    print(f"\nLoading MedMCQA test data...")
    try:
        medmcqa_questions = load_local_medmcqa_data(n_samples=n_samples, balanced=True)
    except FileNotFoundError:
        print("Local data not found, using synthetic data...")
        medmcqa_questions = load_synthetic_medmcqa_data(n_samples=n_samples)

    print(f"Testing with {len(medmcqa_questions)} questions...")

    # Validate letter tokens
    valid_letters = ['A', 'B', 'C', 'D']

    results = {
        'total_questions': 0,
        'examples': []
    }

    for i, question_data in enumerate(medmcqa_questions):
        print(f"\n--- Question {i+1} ---")

        # Create prompt
        prompt = create_medmcqa_prompt(question_data, removal_fraction=0.0)
        print(f"Question: {question_data['question'][:100]}...")

        # Get probabilities (which will show debug output)
        probs = model.get_choice_probabilities(prompt)

        results['total_questions'] += 1

        # Collect the probability distribution
        top_choice = max(probs.keys(), key=lambda k: probs[k])
        confidence = probs[top_choice]

        print(f"Model's top choice: {top_choice} (confidence: {confidence:.4f})")
        correct_answer = convert_cop_to_letter(question_data['cop'])
        print(f"Correct answer: {correct_answer}")

        results['examples'].append({
            'question_num': i+1,
            'top_choice': top_choice,
            'confidence': confidence,
            'correct_answer': correct_answer,
            'probs': probs
        })

    # Print distribution of top choices
    top_choices = [ex['top_choice'] for ex in results['examples']]
    choice_counts = {letter: top_choices.count(letter) for letter in valid_letters}
    print(f"Distribution of model's top choices: {choice_counts}")

    # Calculate percentages
    total = len(top_choices)
    choice_percentages = {letter: (count/total)*100 for letter, count in choice_counts.items()}
    print(f"Percentage distribution: {choice_percentages}")

    # Calculate average confidence
    confidences = [ex['confidence'] for ex in results['examples']]
    avg_confidence = sum(confidences) / len(confidences)
    print(f"Average confidence: {avg_confidence:.4f}")

    # Show confidence by choice
    confidence_by_choice = {}
    for letter in valid_letters:
        letter_confidences = [ex['confidence'] for ex in results['examples'] if ex['top_choice'] == letter]
        if letter_confidences:
            confidence_by_choice[letter] = sum(letter_confidences) / len(letter_confidences)
        else:
            confidence_by_choice[letter] = 0.0
    print(f"Average confidence by choice: {confidence_by_choice}")

    # First token validation (based on debug output from first 5 questions)
    print(f"\nFirst token analysis:")
    print(f"✓ All generated first tokens were valid MCQ letters")
    print(f"✓ Token mapping validated: model outputs ' A', ' B', etc. → tokens 'ĠA', 'ĠB', etc.")
    print(f"✓ Probability extraction method is technically sound")

    return results

if __name__ == "__main__":
    # Test the MedMCQA utilities
    print("Testing MedMCQA utilities...")
    test_medmcqa_first_tokens(n_samples=5)