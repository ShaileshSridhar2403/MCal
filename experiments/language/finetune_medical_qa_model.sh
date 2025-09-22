#!/bin/bash

# Script to launch experiment_utils.py with accelerate and proper environment settings

# Set CUDA_HOME to CONDA_PREFIX
export CUDA_HOME=$CONDA_PREFIX

# Enable DeepSpeed for accelerate
export ACCELERATE_USE_DEEPSPEED=true

# Optional: Set other useful environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings in multi-process

# Print environment info
echo "========================================="
echo "Running experiment with accelerate"
echo "CUDA_HOME: $CUDA_HOME"
echo "ACCELERATE_USE_DEEPSPEED: $ACCELERATE_USE_DEEPSPEED"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "========================================="

# Launch the experiment with accelerate
# Pass all command line arguments to the script
accelerate launch \
    --num_processes=2 \
    --zero_stage=1 \
    experiments/language/finetune_medical_qa_model.py "$@"
