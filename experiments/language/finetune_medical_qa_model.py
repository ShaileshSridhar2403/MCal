#!/usr/bin/env python3
"""
Standalone LoRA Medical QA Training Script - v8 (Manual Masking)

Trains LoRA models on medical QA datasets with specific data schemas.
This version implements precise loss masking by manually creating a 'labels'
column, training the model only on the completion tokens.
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
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# Add experiments to path for all_data_loaders
experiments_root = Path(__file__).parent.parent

from experiments.all_data_loaders import load_medqa_ablated_prob, load_medmcqa_ablated_prob, tokenize_and_mask_medqa, tokenize_and_mask_medmcqa

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    os.environ["NCCL_P2P_DISABLE"] = "0"

    # --- Path and Directory Setup ---
    args.output_dir = str(Path(args.output_dir).expanduser() / args.dataset)

    dataset_output_dir = Path(args.output_dir) / f"{args.dataset}_p{args.p_ablate}"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(dataset_output_dir)
    logger.info(f"All artifacts will be saved to: {args.output_dir}")

    # --- Model and Tokenizer Setup (Load Tokenizer First) ---
    # We need the tokenizer for the data processing step
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Data Loading and Preparation ---
    logger.info(f"Loading '{args.dataset}' using custom data loader...")
    
    if args.dataset == "medqa":
        train_examples = load_medqa_ablated_prob(split='train', p_ablate=args.p_ablate)
        eval_examples = load_medqa_ablated_prob(split='test', p_ablate=args.p_ablate)
    elif args.dataset == "medmcqa":
        train_examples = load_medmcqa_ablated_prob(split='train', p_ablate=args.p_ablate)
        eval_examples = load_medmcqa_ablated_prob(split='test', p_ablate=args.p_ablate)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)
    
    # Use a lambda to pass the dataset type and tokenizer to the formatter
    if args.dataset == "medqa":
        train_dataset = train_dataset.map(lambda x: tokenize_and_mask_medqa(x, tokenizer))
        eval_dataset = eval_dataset.map(lambda x: tokenize_and_mask_medqa(x, tokenizer))
    elif args.dataset == "medmcqa":
        train_dataset = train_dataset.map(lambda x: tokenize_and_mask_medmcqa(x, tokenizer))
        eval_dataset = eval_dataset.map(lambda x: tokenize_and_mask_medmcqa(x, tokenizer))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"Training on {len(train_dataset)} samples, evaluating on {len(eval_dataset)} samples.")

    # --- Model Loading ---
    logger.info(f"Loading base model: {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # --- PEFT and LoRA Configuration ---
    logger.info("Setting up LoRA configuration...")
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training Configuration ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        do_eval=True,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mean_token_accuracy",
        greater_is_better=True,
        remove_unused_columns=True,
        bf16=True,
        max_length=args.max_length,
        completion_only_loss=False,
        max_grad_norm=1.0,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # --- Train the Model ---
    logger.info("Starting training...")
    trainer.train()

    # --- Save Artifacts ---
    logger.info("Saving final LoRA adapters...")
    trainer.save_model()

    merged_output_dir = Path(args.output_dir) / "merged_model"
    merged_output_dir.mkdir(exist_ok=True)

    # Merge LoRA weights into the base model and unload adapters
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_output_dir))
    tokenizer.save_pretrained(str(merged_output_dir))
    logger.info(f"Merged model saved to {merged_output_dir}")

    config_info = vars(args)
    config_file = Path(args.output_dir) / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=2)

    logger.info("Training completed successfully!")
    logger.info(f"LoRA adapters saved to: {args.output_dir}")
    logger.info("To merge the model, run a separate script after training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone LoRA training for medical QA (Manual Masking)")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, choices=["medqa", "medmcqa"], required=True)
    parser.add_argument("--p_ablate", type=float, default=0.0)
    
    # Model args
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent.parent.parent / "saved_models"))
    
    # Training args
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()
    main(args)

