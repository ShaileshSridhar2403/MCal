#!/usr/bin/env python3
"""
Standalone QLoRA Medical QA Training Script

Trains QLora models on ablated medical QA datasets using only all_data_loaders.py.
Completely self-contained implementation without dependencies on qlora_utils.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import torch
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# Add experiments to path for all_data_loaders
experiments_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_root))

from all_data_loaders import load_medqa_ablated_prob, load_medmcqa_ablated_prob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_medqa_to_training_format(texts, labels):
    """Convert MedQA data to training format (5 options: A-E)."""
    training_data = []
    option_letters = ['A', 'B', 'C', 'D', 'E']

    for (question, options), label in zip(texts, labels):
        # Build options text
        options_text = ""
        for letter in option_letters:
            if letter in options:
                options_text += f"{letter}. {options[letter]}\n"

        # Create input-output pair with clear boundary for completion_only_loss
        input_text = f"Question: {question}\n\nOptions:\n{options_text}\nAnswer:"
        correct_letter = option_letters[label.item()]
        full_text = f"{input_text} {correct_letter}"

        # Use the format that TRL expects for completion_only_loss
        training_data.append({
            "text": full_text,
            "prompt": input_text,  # Everything before the answer
            "completion": f" {correct_letter}"  # Just the answer part
        })

    return training_data

def convert_medmcqa_to_training_format(texts, labels):
    """Convert MedMCQA data to training format (4 options: A-D)."""
    training_data = []
    option_letters = ['A', 'B', 'C', 'D']

    for (question, options), label in zip(texts, labels):
        # Build options text
        options_text = ""
        for letter in option_letters:
            if letter in options:
                options_text += f"{letter}. {options[letter]}\n"

        # Create input-output pair with clear boundary for completion_only_loss
        input_text = f"Question: {question}\n\nOptions:\n{options_text}\nAnswer:"
        correct_letter = option_letters[label.item()]
        full_text = f"{input_text} {correct_letter}"

        # Use the format that TRL expects for completion_only_loss
        training_data.append({
            "text": full_text,
            "prompt": input_text,  # Everything before the answer
            "completion": f" {correct_letter}"  # Just the answer part
        })

    return training_data


def main(args):

    # Expand paths
    args.base_model = str(Path(args.base_model).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())

    # Create dataset-specific output directory
    dataset_output_dir = Path(args.output_dir) / f"{args.dataset}_p{args.p_ablate}_n{args.n_samples}"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(dataset_output_dir)

    logger.info("Starting standalone QLoRA Medical QA training")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Ablation probability: {args.p_ablate}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output directory: {args.output_dir}")

    # Step 1: Load and convert training data
    logger.info("Loading training data from all_data_loaders...")
    if args.dataset == "medqa":
        train_texts, train_labels = load_medqa_ablated_prob(
            split='train',
            p_ablate=args.p_ablate,
            n_samples=args.n_samples
        )
        train_examples = convert_medqa_to_training_format(train_texts, train_labels)
    elif args.dataset == "medmcqa":
        train_texts, train_labels = load_medmcqa_ablated_prob(
            split='train',
            p_ablate=args.p_ablate,
            n_samples=args.n_samples
        )
        train_examples = convert_medmcqa_to_training_format(train_texts, train_labels)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"Loaded {len(train_examples)} questions")


    # Step 2: Setup model and tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    logger.info("Preparing model for training...")
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    logger.info("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 3: Prepare datasets for SFTTrainer
    logger.info("Preparing training dataset...")
    train_dataset = Dataset.from_list(train_examples)

    # Step 3b: Prepare evaluation dataset
    logger.info("Preparing evaluation dataset...")
    if args.dataset == "medqa":
        eval_texts, eval_labels = load_medqa_ablated_prob(
            split='test',
            p_ablate=args.p_ablate,
            n_samples=args.n_eval_samples
        )
        eval_examples = convert_medqa_to_training_format(eval_texts, eval_labels)
    elif args.dataset == "medmcqa":
        eval_texts, eval_labels = load_medmcqa_ablated_prob(
            split='test',
            p_ablate=args.p_ablate,
            n_samples=args.n_eval_samples
        )
        eval_examples = convert_medmcqa_to_training_format(eval_texts, eval_labels)

    eval_dataset = Dataset.from_list(eval_examples)

    # Step 4: Setup SFT training configuration
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=1e-2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,
        report_to="none",
        max_length=args.max_length,
        completion_only_loss=True
    )

    # Step 5: Create SFTTrainer with response template
    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting training...")
    trainer.train()

    # Step 7: Save model
    logger.info("Saving LoRA adapters...")
    trainer.save_model()

    # Step 8: Save merged model
    merged_output_dir = Path(args.output_dir) / "merged_model"
    merged_output_dir.mkdir(exist_ok=True)

    logger.info("Merging LoRA adapters with base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_output_dir))
    tokenizer.save_pretrained(str(merged_output_dir))
    logger.info(f"Merged model saved to {merged_output_dir}")

    # Step 9: Save configuration info
    config_info = vars(args)  # Convert argparse Namespace to dict

    config_file = Path(args.output_dir) / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=2)

    logger.info("Training completed successfully!")
    logger.info(f"LoRA adapters saved to: {args.output_dir}")
    logger.info(f"Merged model saved to: {args.output_dir}/merged_model")
    logger.info(f"Training config saved to: {config_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone QLoRA training for medical QA")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, choices=["medqa", "medmcqa"], required=True,
                       help="Dataset to use for training")
    parser.add_argument("--n_samples", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--p_ablate", type=float, default=0.5,
                       help="Probability of ablating each token (0.0 to 1.0)")

    # Model arguments
    parser.add_argument("--base_model", type=str,
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Path to base LLaMA model or HuggingFace model ID")
    parser.add_argument("--output_dir", type=str,
                       default=str(Path(__file__).parent.parent.parent / "experiments" / "language" / "qlora_medical_qa_models"),
                       help="Directory to save trained models")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")

    # Evaluation arguments
    parser.add_argument("--n_eval_samples", type=int, default=1000,
                       help="Number of test samples for evaluation")


    args = parser.parse_args()
    main(args)