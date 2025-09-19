# QLoRA Ablation Training Plan for KL Divergence Analysis

## Overview

Implement QLoRA (Quantized Low-Rank Adaptation) fine-tuning on ablated medical question-answering data to study how training on corrupted inputs affects model predictions and calibration. The goal is to measure KL divergence between QLoRA-trained models and uniform distribution.

## Background

- **QLoRA**: Efficient fine-tuning using 4-bit quantization + LoRA adapters
- **Ablation Strategy**: Train on inputs with tokens randomly removed (binomial sampling)
- **Analysis**: Compare KL divergence before/after QLoRA training on clean vs ablated data

## Technical Approach

### 1. Data Preparation

#### 1.1 Ablated Dataset Generation
```python
def create_ablated_training_data(questions, ablation_rate=0.3):
    """
    Create training dataset with binomial token removal

    Args:
        questions: Original MedQA/MedMCQA questions
        ablation_rate: Probability of removing each token (0.0-1.0)

    Returns:
        List of (ablated_question, original_answer) pairs
    """
```

**Strategy Options:**
- **Binomial Token Removal**: Each token removed with probability `p`
- **Fixed Fraction**: Remove exactly `X%` of tokens randomly
- **Content-Only**: Only remove question content tokens (preserve structure)

#### 1.2 Training Data Format
```json
{
    "instruction": "Answer the following medical question:",
    "input": "Question: [ABLATED_QUESTION]\nA. Option1\nB. Option2\nC. Option3\nD. Option4",
    "output": "A"
}
```

### 2. QLoRA Implementation

#### 2.1 Model Setup
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def setup_qlora_model(base_model_path, lora_config):
    """
    Setup 4-bit quantized model with LoRA adapters

    Components:
    - 4-bit quantization (BitsAndBytesConfig)
    - LoRA adapters on attention layers
    - Gradient checkpointing for memory efficiency
    """
```

#### 2.2 LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,                          # Low-rank dimension
    lora_alpha=32,                 # LoRA scaling parameter
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

#### 2.3 Training Configuration
```python
training_args = TrainingArguments(
    output_dir="./qlora_ablated_models",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)
```

### 3. Training Pipeline

#### 3.1 Single Model with Binomial Ablation Distribution
Train a single QLoRA model on training data where each example has a randomly sampled ablation level:
- **Ablation Sampling**: Each training example gets ablation rate sampled from binomial distribution
- **Distribution Parameters**: e.g., p=0.3 for token removal probability
- **Diverse Training**: Model sees examples across full spectrum of corruption levels (0-90%)

#### 3.2 Training Data Generation
```python
def create_binomial_ablated_training_data(questions, p_remove_dist=(0.0, 0.9)):
    """
    Create training dataset where each example has randomly sampled ablation

    For each question:
    1. Sample ablation rate from binomial/uniform distribution
    2. Apply that ablation rate to create corrupted input
    3. Keep original answer as target

    Args:
        questions: Original training questions
        p_remove_dist: (min, max) ablation rate range

    Returns:
        Single dataset with mixed ablation levels
    """
```

### 4. Integration with MCal Framework

#### 4.1 QLoRA Model Wrapper
```python
class MCal_QLoRA_Model:
    """QLoRA model wrapper for MCal integration"""

    def __init__(self, base_model_path, lora_adapter_path):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        self.model = PeftModel.from_pretrained(self.base_model, lora_adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    def get_choice_probabilities(self, prompt, num_options=5):
        """Extract A,B,C,D,E probabilities - same interface as MCal_LLaMAModel"""
```

#### 4.2 Benchmark Integration
```python
def load_medqa_data_qlora(n_samples=10, n_fractions=10):
    """
    Load single QLoRA model trained on binomially ablated data

    Args:
        n_samples: Number of test samples
        n_fractions: Number of test ablation fractions

    Returns:
        predictions, labels (same format as existing methods)
    """
```

### 5. Experimental Design

#### 5.1 Comparison Matrix
| Method | Training Data | Test Data | Purpose |
|--------|---------------|-----------|---------|
| **Original** | Clean | Clean | Baseline performance |
| **QLoRA-Clean** | Clean | Clean | QLoRA fine-tuning effect |
| **QLoRA-Binomial** | Binomially ablated (0-90%) | Clean | Corruption robustness |
| **QLoRA-Binomial** | Binomially ablated (0-90%) | Various ablation levels | Cross-corruption performance |

#### 5.2 Evaluation Metrics
- **KL Divergence**: From uniform distribution (primary metric)
- **Accuracy**: Multiple-choice accuracy on clean test data
- **Calibration**: Reliability diagrams, ECE
- **Robustness**: Performance on ablated test inputs

### 6. Implementation Structure

#### 6.1 New Files to Create
```
MCal/experiments/language/
├── qlora_utils.py                    # QLoRA training utilities
├── qlora_ablation_trainer.py         # Training script
├── medqa_qlora_benchmark.py          # QLoRA benchmark integration
├── medmcqa_qlora_benchmark.py        # QLoRA benchmark for MedMCQA
└── configs/
    ├── qlora_config.yaml             # Training configurations
    └── ablation_config.yaml          # Ablation strategies
```

#### 6.2 Key Components

**qlora_utils.py:**
```python
def create_binomial_ablated_dataset(questions, p_remove=0.3)
def setup_qlora_training(base_model_path, lora_config)
def train_qlora_model(model, dataset, training_args)
class MCal_QLoRA_Model(MCal_LLaMAModel)  # Inherit from base
```

**qlora_ablation_trainer.py:**
```python
def main():
    # Train multiple QLoRA models for different ablation rates
    for ablation_rate in [0.0, 0.1, 0.3, 0.5, 0.7]:
        train_and_save_qlora_model(ablation_rate)
```

**medqa_qlora_benchmark.py:**
```python
def load_medqa_data_qlora(ablation_rate, n_samples, n_fractions):
    # Load QLoRA model and run standard KL benchmark

def process_medqa_qlora_dataset(ablation_rates, methods, ...):
    # Compare QLoRA models with different ablation training
```

### 7. Expected Outcomes

#### 7.1 Hypotheses
1. **Robustness**: Models trained on ablated data should be more robust to input corruption
2. **Calibration**: QLoRA training might improve or degrade calibration depending on ablation level
3. **KL Divergence**: Heavy ablation training might push predictions closer to uniform (lower KL)

#### 7.2 Analysis Dimensions
- **Ablation Rate vs KL Divergence**: How training corruption affects calibration
- **Cross-Ablation Performance**: Models trained on X% corruption tested on Y% corruption
- **Calibration Methods**: How MCal, temperature scaling work on QLoRA models

### 8. Resource Requirements

#### 8.1 Computational
- **GPU Memory**: ~24GB for 8B model with QLoRA (vs ~40GB for full fine-tuning)
- **Training Time**: ~2-4 hours per ablation level on A100
- **Storage**: ~500MB per LoRA adapter (vs ~16GB for full model)

#### 8.2 Data Requirements
- **Training Samples**: 1000-5000 questions per ablation level
- **Validation**: 500-1000 questions for early stopping
- **Test**: Existing balanced dev sets for evaluation

### 9. Implementation Phases

#### Phase 1: Core QLoRA Integration (Week 1)
- [ ] Implement `qlora_utils.py` with basic QLoRA setup
- [ ] Create ablated dataset generation
- [ ] Test QLoRA training on small dataset

#### Phase 2: Training Pipeline (Week 2)
- [ ] Implement multi-ablation training script
- [ ] Train QLoRA models for all ablation rates
- [ ] Validate model outputs and sanity checks

#### Phase 3: MCal Integration (Week 3)
- [ ] Create `MCal_QLoRA_Model` wrapper
- [ ] Integrate with existing benchmark framework
- [ ] Test KL divergence measurements

#### Phase 4: Analysis & Results (Week 4)
- [ ] Run comprehensive benchmarks
- [ ] Compare with existing methods (attention mask, token drop)
- [ ] Generate analysis reports and visualizations

### 10. Usage Example

```bash
# Train QLoRA models
python qlora_ablation_trainer.py --dataset medqa --ablation_rates 0.0,0.1,0.3,0.5,0.7

# Run QLoRA benchmark
python medqa_qlora_benchmark.py --ablation_rates medium,heavy --methods baseline,mcal_ce --samples 100

# Compare all methods
python medqa_kl_benchmark.py --methods baseline,attention_mask,token_drop,qlora_medium --samples 100
```

This plan provides a comprehensive approach to integrating QLoRA ablation training into the MCal framework while maintaining consistency with existing methods and enabling comparative analysis of different corruption strategies.