#!/usr/bin/env python3
"""
QLoRA Ablation Training Script

Trains a single QLoRA model on binomially ablated medical QA data (0-90% ablation range)
for studying corruption-aware training effects on model calibration.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import logging

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))

from qlora_utils import (
    create_binomial_ablated_dataset,
    setup_qlora_model,
    prepare_training_data,
    get_training_arguments,
    train_qlora_model,
    save_training_info,
    load_local_data_for_training
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train QLoRA model on binomially ablated medical QA data")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, choices=["medqa", "medmcqa"], required=True,
                       help="Dataset to use for training")
    parser.add_argument("--n_samples", type=int, default=1000,
                       help="Number of training samples")
    parser.add_argument("--balanced", action="store_true", default=True,
                       help="Use balanced answer distribution")

    # Ablation arguments
    parser.add_argument("--ablation_min", type=float, default=0.0,
                       help="Minimum ablation rate")
    parser.add_argument("--ablation_max", type=float, default=0.9,
                       help="Maximum ablation rate")
    parser.add_argument("--preserve_structure", action="store_true", default=True,
                       help="Preserve structural elements during ablation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    # Model arguments
    parser.add_argument("--base_model", type=str,
                       default="~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/",
                       help="Path to base LLaMA model")
    parser.add_argument("--output_dir", type=str,
                       default="./qlora_binomial_models",
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

    # Utility arguments
    parser.add_argument("--dry_run", action="store_true",
                       help="Run data preparation only, don't train")

    args = parser.parse_args()

    # Expand paths
    args.base_model = str(Path(args.base_model).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting QLoRA binomial ablation training")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Ablation range: {args.ablation_min}-{args.ablation_max}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output directory: {args.output_dir}")

    # Step 1: Load training data
    logger.info("Loading training data...")
    questions = load_local_data_for_training(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        balanced=args.balanced
    )
    logger.info(f"Loaded {len(questions)} questions")

    # Step 2: Create binomially ablated training dataset
    logger.info("Creating binomially ablated training dataset...")
    training_data = create_binomial_ablated_dataset(
        questions=questions,
        p_remove_range=(args.ablation_min, args.ablation_max),
        preserve_structure=args.preserve_structure,
        seed=args.seed
    )

    if args.dry_run:
        logger.info("Dry run completed. Exiting without training.")

        # Save some examples for inspection
        examples_file = Path(args.output_dir) / "dry_run_examples.json"
        import json
        with open(examples_file, 'w') as f:
            json.dump(training_data[:20], f, indent=2)
        logger.info(f"Sample examples saved to {examples_file}")
        return

    # Step 3: Setup QLoRA model
    logger.info("Setting up QLoRA model...")
    model, tokenizer = setup_qlora_model(args.base_model)

    # Step 4: Prepare training data
    logger.info("Preparing training data...")
    training_dataset = prepare_training_data(
        training_data=training_data,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    # Step 5: Setup training arguments
    training_args = get_training_arguments(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Step 6: Train the model
    logger.info("Starting training...")
    trained_model = train_qlora_model(
        model=model,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        training_args=training_args
    )

    # Step 7: Save training information
    logger.info("Saving training information...")
    config_info = {
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "ablation_range": (args.ablation_min, args.ablation_max),
        "preserve_structure": args.preserve_structure,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "base_model": args.base_model,
        "seed": args.seed
    }

    save_training_info(
        output_dir=args.output_dir,
        ablation_range=(args.ablation_min, args.ablation_max),
        training_data=training_data,
        config_info=config_info
    )

    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"To use this model, load it with MCal_QLoRA_Model('{args.base_model}', '{args.output_dir}')")

if __name__ == "__main__":
    main()