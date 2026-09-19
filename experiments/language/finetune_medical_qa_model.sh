#!/bin/bash

# Script to launch experiment_utils.py with accelerate and proper environment settings

# Set CUDA_HOME to CONDA_PREFIX
export CUDA_HOME=$CONDA_PREFIX

# Enable DeepSpeed for accelerate
export ACCELERATE_USE_DEEPSPEED=true

# Optional: Set other useful environment variables
# Run from the repository root so experiments.* and mcal.* import
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings in multi-process

# Print environment info
echo "========================================="
echo "Running experiment with accelerate"
echo "CUDA_HOME: $CUDA_HOME"
echo "ACCELERATE_USE_DEEPSPEED: $ACCELERATE_USE_DEEPSPEED"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "========================================="

# Detect number of CUDA devices
if command -v python3 &> /dev/null; then
    NUM_CUDA_DEVICES=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
else
    NUM_CUDA_DEVICES=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
fi

# Fallback to 1 if no CUDA devices or detection fails
if [ -z "$NUM_CUDA_DEVICES" ] || [ "$NUM_CUDA_DEVICES" -eq 0 ]; then
    NUM_CUDA_DEVICES=1
    echo "No CUDA devices detected, using CPU with 1 process"
else
    echo "Detected $NUM_CUDA_DEVICES CUDA device(s)"
fi

# Launch the experiment with accelerate
# Pass all command line arguments to the script
accelerate launch \
    --num_processes=$NUM_CUDA_DEVICES \
    --zero_stage=1 \
    experiments/language/finetune_medical_qa_model.py "$@"
