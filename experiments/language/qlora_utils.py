#!/usr/bin/env python3
"""
QLoRA Utilities for MCal Language Experiments

Implements QLoRA (Quantized Low-Rank Adaptation) training on binomially ablated medical QA data
for studying the effects of corruption-aware training on model calibration.
"""

import torch
import numpy as np
import random
import json
import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from tqdm import tqdm
from datasets import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== BINOMIAL ABLATION DATASET GENERATION =====

def create_binomial_ablated_dataset(questions, p_remove_range=(0.0, 0.9), preserve_structure=True, seed=42):
    """
    Create training dataset with binomially sampled ablation rates.

    Args:
        questions: List of question dictionaries (MedQA/MedMCQA format)
        p_remove_range: (min, max) range for sampling ablation rates
        preserve_structure: If True, preserve "Question:", "A.", etc.
        seed: Random seed for reproducibility

    Returns:
        List of training examples with varied ablation levels
    """
    random.seed(seed)
    np.random.seed(seed)

    training_data = []
    min_ablation, max_ablation = p_remove_range

    logger.info(f"Creating binomial ablated dataset with ablation range {p_remove_range}")
    logger.info(f"Preserve structure: {preserve_structure}")

    for question_data in tqdm(questions, desc="Creating binomially ablated training data"):
        # Sample ablation rate for this example
        p_remove = random.uniform(min_ablation, max_ablation)

        # Create the full prompt
        if 'options' in question_data or 'opa' in question_data:
            # Determine format and create prompt
            if 'options' in question_data:  # MedQA format
                prompt = create_medqa_training_prompt(question_data)
                answer_key = question_data.get('answer_idx', 'A')
            else:  # MedMCQA format
                prompt = create_medmcqa_training_prompt(question_data)
                # Convert 1-indexed to letter
                cop = question_data.get('cop', 1)
                answer_key = chr(ord('A') + cop - 1)
        else:
            logger.warning(f"Unknown question format: {question_data.keys()}")
            continue

        # Apply binomial ablation to the question content
        ablated_prompt = apply_binomial_ablation(prompt, p_remove, preserve_structure)

        # Create training pair
        training_data.append({
            'input': ablated_prompt,
            'output': answer_key,
            'original_prompt': prompt,
            'ablation_rate': p_remove
        })

    logger.info(f"Created {len(training_data)} binomially ablated training examples")

    # Log ablation rate distribution
    ablation_rates = [example['ablation_rate'] for example in training_data]
    logger.info(f"Ablation rate stats: mean={np.mean(ablation_rates):.3f}, "
                f"std={np.std(ablation_rates):.3f}, "
                f"min={np.min(ablation_rates):.3f}, "
                f"max={np.max(ablation_rates):.3f}")

    return training_data

def apply_binomial_ablation(text, p_remove, preserve_structure=True):
    """
    Apply binomial token removal to text.

    Args:
        text: Input text to ablate
        p_remove: Probability of removing each token
        preserve_structure: If True, preserve structural elements

    Returns:
        Ablated text with tokens randomly removed
    """
    if p_remove <= 0:
        return text

    # Split into lines to preserve overall structure
    lines = text.split('\n')
    ablated_lines = []

    for line in lines:
        if preserve_structure and any(marker in line for marker in ['Question:', 'A.', 'B.', 'C.', 'D.', 'E.', 'Answer:']):
            # For structural lines, only ablate the content after the marker
            if ':' in line:
                parts = line.split(':', 1)
                separator = ':'
            elif '.' in line and line.strip().split('.')[0] in ['A', 'B', 'C', 'D', 'E']:
                parts = line.split('.', 1)
                separator = '.'
            else:
                # No clear separator, ablate whole line
                ablated_lines.append(ablate_tokens(line, p_remove))
                continue

            if len(parts) == 2:
                marker, content = parts
                ablated_content = ablate_tokens(content.strip(), p_remove)
                ablated_lines.append(f"{marker}{separator} {ablated_content}")
            else:
                ablated_lines.append(line)
        else:
            # Ablate the entire line
            ablated_lines.append(ablate_tokens(line, p_remove))

    return '\n'.join(ablated_lines)

def ablate_tokens(text, p_remove):
    """Apply binomial removal to tokens in text."""
    if not text.strip():
        return text

    tokens = text.split()
    kept_tokens = []

    for token in tokens:
        if random.random() > p_remove:  # Keep token
            kept_tokens.append(token)
        # Otherwise, remove token (skip)

    return ' '.join(kept_tokens) if kept_tokens else ""

def create_medqa_training_prompt(question_data):
    """Create training prompt for MedQA format."""
    question = question_data['question']
    options = question_data.get('options', {})

    prompt = f"Question: {question}\n"

    for key in sorted(options.keys()):
        prompt += f"{key}. {options[key]}\n"

    prompt += "Answer:"
    return prompt

def create_medmcqa_training_prompt(question_data):
    """Create training prompt for MedMCQA format."""
    question = question_data['question']

    prompt = f"Question: {question}\n"
    prompt += f"A. {question_data.get('opa', '')}\n"
    prompt += f"B. {question_data.get('opb', '')}\n"
    prompt += f"C. {question_data.get('opc', '')}\n"
    prompt += f"D. {question_data.get('opd', '')}\n"
    prompt += "Answer:"

    return prompt

# ===== QLORA MODEL SETUP =====

def get_qlora_config():
    """Get default QLoRA configuration."""
    return LoraConfig(
        r=16,                          # Low-rank dimension
        lora_alpha=32,                 # LoRA scaling parameter
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

def get_quantization_config():
    """Get 4-bit quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def setup_qlora_model(base_model_path, lora_config=None, quantization_config=None):
    """
    Setup 4-bit quantized model with LoRA adapters.

    Args:
        base_model_path: Path to base LLaMA model
        lora_config: LoRA configuration (uses default if None)
        quantization_config: Quantization configuration (uses default if None)

    Returns:
        model, tokenizer
    """
    if lora_config is None:
        lora_config = get_qlora_config()

    if quantization_config is None:
        quantization_config = get_quantization_config()

    logger.info(f"Loading base model from: {base_model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    logger.info("QLoRA model setup complete")
    logger.info(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    logger.info(f"Total parameters: {model.num_parameters():,}")

    return model, tokenizer

# ===== TRAINING UTILITIES =====

def prepare_training_data(training_data, tokenizer, max_length=512):
    """
    Prepare training data for QLoRA fine-tuning.

    Args:
        training_data: List of training examples from create_binomial_ablated_dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Hugging Face Dataset ready for training
    """
    def format_example(example):
        """Format example for instruction following."""
        instruction = "Answer the following medical question by selecting the correct option (A, B, C, D, or E)."
        input_text = f"### Instruction:\n{instruction}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        return {"text": input_text}

    # Format all examples
    formatted_data = [format_example(example) for example in training_data]

    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

def get_training_arguments(output_dir, num_epochs=3, batch_size=4, learning_rate=1e-4):
    """Get training arguments for QLoRA fine-tuning."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        run_name=f"qlora-binomial-ablation",
        optim="paged_adamw_8bit",  # Memory efficient optimizer
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

# ===== MCAL INTEGRATION =====

class MCal_QLoRA_Model:
    """QLoRA model wrapper for MCal integration."""

    def __init__(self, base_model_path, lora_adapter_path, device=None):
        """
        Initialize QLoRA model for MCal framework.

        Args:
            base_model_path: Path to base LLaMA model
            lora_adapter_path: Path to trained LoRA adapters
            device: Device to load model on
        """
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading QLoRA model from {base_model_path} with adapters {lora_adapter_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model with quantization
        quantization_config = get_quantization_config()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)

        logger.info("✓ QLoRA model loaded successfully")

    def get_choice_probabilities(self, prompt, num_options=5):
        """Extract probabilities for A,B,C,D,E choices - same interface as MCal_LLaMAModel."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            # Get logits of the last token
            outputs = self.model(input_ids)
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

    def get_choice_probabilities_with_attention_mask(self, prompt, attention_mask=None, num_options=5):
        """Extract probabilities using custom attention mask - for compatibility."""
        # For QLoRA models, we don't apply additional attention masking during inference
        # since the model was already trained on ablated data
        return self.get_choice_probabilities(prompt, num_options)

    def forward(self, prompts, num_options=5):
        """Process batch of prompts and return probabilities."""
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            probs = self.get_choice_probabilities(prompt, num_options)
            results.append(probs)

        return results

# ===== TRAINING ORCHESTRATION =====

def train_qlora_model(model, tokenizer, training_dataset, training_args):
    """
    Train QLoRA model on binomially ablated data.

    Args:
        model: QLoRA model setup with adapters
        tokenizer: HuggingFace tokenizer
        training_dataset: Prepared training dataset
        training_args: Training arguments

    Returns:
        Trained model
    """
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting QLoRA training on binomially ablated data...")
    trainer.train()

    # Save the LoRA adapters
    trainer.save_model()
    logger.info(f"LoRA adapters saved to {training_args.output_dir}")

    # Also save the merged model
    merged_output_dir = Path(training_args.output_dir) / "merged_model"
    merged_output_dir.mkdir(exist_ok=True)

    logger.info("Merging LoRA adapters with base model...")
    try:
        # Merge and unload the adapters
        merged_model = model.merge_and_unload()

        # Save the merged model
        merged_model.save_pretrained(str(merged_output_dir))
        tokenizer.save_pretrained(str(merged_output_dir))

        logger.info(f"Merged model saved to {merged_output_dir}")
        logger.info("Note: The merged model is ready for standalone deployment")

    except Exception as e:
        logger.warning(f"Failed to save merged model: {e}")
        logger.warning("Continuing with adapter-only saving")

    return model

def save_training_info(output_dir, ablation_range, training_data, config_info):
    """Save training metadata and configuration."""
    info = {
        "ablation_range": ablation_range,
        "num_training_examples": len(training_data),
        "ablation_stats": {
            "mean": float(np.mean([ex['ablation_rate'] for ex in training_data])),
            "std": float(np.std([ex['ablation_rate'] for ex in training_data])),
            "min": float(np.min([ex['ablation_rate'] for ex in training_data])),
            "max": float(np.max([ex['ablation_rate'] for ex in training_data]))
        },
        "config": config_info,
        "model_type": "qlora_binomial_ablated",
        "saved_formats": {
            "lora_adapters": "Base directory contains LoRA adapters",
            "merged_model": "merged_model/ subdirectory contains standalone merged model"
        },
        "usage_notes": {
            "adapter_model": "Use MCal_QLoRA_Model(base_model_path, adapter_path)",
            "merged_model": "Use MCal_Merged_QLoRA_Model(merged_model_path)"
        }
    }

    with open(Path(output_dir) / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    # Save a few example training instances for inspection
    examples = training_data[:10]
    with open(Path(output_dir) / "training_examples.json", 'w') as f:
        json.dump(examples, f, indent=2)

# ===== UTILITY FUNCTIONS =====

class MCal_Merged_QLoRA_Model:
    """Merged QLoRA model wrapper for MCal integration (standalone model)."""

    def __init__(self, merged_model_path, device=None):
        """
        Initialize merged QLoRA model for MCal framework.

        Args:
            merged_model_path: Path to merged model directory
            device: Device to load model on
        """
        self.merged_model_path = merged_model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading merged QLoRA model from {merged_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load merged model (no quantization needed since it's already merged)
        self.model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use float16 for efficiency
        )

        logger.info("✓ Merged QLoRA model loaded successfully")

    def get_choice_probabilities(self, prompt, num_options=5):
        """Extract probabilities for A,B,C,D,E choices - same interface as MCal_LLaMAModel."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            # Get logits of the last token
            outputs = self.model(input_ids)
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

    def get_choice_probabilities_with_attention_mask(self, prompt, attention_mask=None, num_options=5):
        """Extract probabilities using custom attention mask - for compatibility."""
        # For merged models, we don't apply additional attention masking during inference
        # since the model was already trained on ablated data
        return self.get_choice_probabilities(prompt, num_options)

    def forward(self, prompts, num_options=5):
        """Process batch of prompts and return probabilities."""
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            probs = self.get_choice_probabilities(prompt, num_options)
            results.append(probs)

        return results

def load_local_data_for_training(dataset_name, n_samples=1000, balanced=True):
    """
    Load local dataset for QLoRA training.

    Args:
        dataset_name: "medqa" or "medmcqa"
        n_samples: Number of training samples
        balanced: Whether to balance answer distribution

    Returns:
        List of question dictionaries ready for ablation
    """
    if dataset_name == "medqa":
        from medqa_utils import load_local_medqa_data
        return load_local_medqa_data(n_samples, balanced=balanced)
    elif dataset_name == "medmcqa":
        from medmcqa_utils import load_local_medmcqa_data
        return load_local_medmcqa_data(n_samples, balanced=balanced)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def map_probs_to_list(prob_dict, num_options=5):
    """Convert probability dict to list format for MCal compatibility."""
    if num_options == 4:
        letters = ['A', 'B', 'C', 'D']
    elif num_options == 5:
        letters = ['A', 'B', 'C', 'D', 'E']
    else:
        raise ValueError(f"Unsupported number of options: {num_options}")

    return [prob_dict.get(letter, 0.0) for letter in letters]