#!/usr/bin/env python3
"""
Test different MCal calibration methods from src/calibrators/mcal_test.py
on an imbalanced dataset with A/B/C classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from typing import Tuple, Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.calibrators.mcal_test import MCal_Test, MCal_CE
from experiment_utils import kl_divergence, missingness_bias

def create_imbalanced_dataset(num_samples: int = 2000, num_features: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a more challenging imbalanced dataset.

    The classification depends on non-linear interactions between features,
    making it harder for a simple MLP to achieve perfect accuracy.

    Class 0: "A" [>0.5, >0.5,  ~0,    ~0, noise...] (Rare)
    Class 1: "B" [ ~0,    ~0, >0.5, >0.5, noise...] (Rare)
    Class 2: "C" [rand, rand, rand, rand, noise...] (Common Distractor)
    
    Args:
        num_samples: Number of samples to generate.
        num_features: Total number of features.
        
    Returns:
        X: Features tensor of shape (num_samples, num_features)
        y: Labels tensor of shape (num_samples,)
    """
    # Define class distribution
    class_probs = torch.tensor([0.1, 0.1, 0.80])
    y = torch.multinomial(class_probs, num_samples, replacement=True)
    
    # Initialize features with noise
    X = torch.rand(num_samples, num_features, dtype=torch.float32) * 0.5
    
    # Class 0: "A" - requires two features to be high
    mask_a = (y == 0)
    X[mask_a, 0] = 0.75 + torch.rand(mask_a.sum()) * 0.25 # f0 is high
    X[mask_a, 1] = 0.75 + torch.rand(mask_a.sum()) * 0.25 # f1 is high

    # Class 1: "B" - requires two other features to be high
    mask_b = (y == 1)
    X[mask_b, 2] = 0.75 + torch.rand(mask_b.sum()) * 0.25 # f2 is high
    X[mask_b, 3] = 0.75 + torch.rand(mask_b.sum()) * 0.25 # f3 is high

    # Class 2: "C" (Distractor) - The first 4 features are noisy and can
    # partially mimic A or B, but not the combination.
    mask_c = (y == 2)
    X[mask_c, :4] = torch.rand(mask_c.sum(), 4)
    
    return X, y


class BaseModel(nn.Module):
    """Simple feedforward neural network for classification."""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_base_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                     epochs: int = 200, lr: float = 0.01) -> None:
    """Train a model using cross-entropy loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            acc = (outputs.argmax(dim=1) == y).float().mean()
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")


def ablate_data(X: torch.Tensor, ablation_prob: float = 0.5) -> torch.Tensor:
    """
    Randomly ablate features by setting them to zero.
    
    Args:
        X: Input features
        ablation_prob: Probability of ablating each feature
        
    Returns:
        Ablated features
    """
    X_ablated = X.clone()
    # Only ablate the first 4 features (the pattern features)
    mask = (torch.rand(X.shape[0], 4) > ablation_prob).float()
    X_ablated[:, :4] *= mask
    return X_ablated


def get_probabilities(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Get probability predictions from model."""
    with torch.no_grad():
        logits = model(X)
        return F.softmax(logits, dim=1)


def train_and_evaluate_calibrators(base_model: nn.Module, 
                                 X_train: torch.Tensor, y_train: torch.Tensor,
                                 X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, Any]:
    """Train and evaluate different MCal calibrators."""
    
    # Generate clean and ablated predictions for training
    clean_probs_train = get_probabilities(base_model, X_train)
    ablated_X_train = ablate_data(X_train)
    ablated_probs_train = get_probabilities(base_model, ablated_X_train)
    
    # Generate clean and ablated predictions for testing
    clean_probs_test = get_probabilities(base_model, X_test)
    ablated_X_test = ablate_data(X_test)
    ablated_probs_test = get_probabilities(base_model, ablated_X_test)
    
    # Device - force CPU to avoid CUDA issues
    device = torch.device('cpu')
    
    # Define calibrators to test
    calibrators = {
        'MCal_Test_KL_Exp': MCal_Test(num_classes=3, loss_mode='KL_Exp'),
        'MCal_Test_Exp_KL': MCal_Test(num_classes=3, loss_mode='Exp_KL'),
        'MCal_Test_CE': MCal_Test(num_classes=3, loss_mode='CE'),
        'MCal_CE_linear': MCal_CE(num_classes=3, head_type='linear'),
        'MCal_CE_mlp': MCal_CE(num_classes=3, head_type='mlp'),
    }
    
    # Move everything to device
    base_model = base_model.to(device)
    ablated_probs_train = ablated_probs_train.to(device)
    clean_probs_train = clean_probs_train.to(device)
    ablated_probs_test = ablated_probs_test.to(device)
    clean_probs_test = clean_probs_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    results = {}
    
    print("\nTraining calibrators...")
    print("="*70)
    
    for name, calibrator in calibrators.items():
        print(f"\nTraining {name}...", end='', flush=True)
        calibrator = calibrator.to(device)
        
        # Train the calibrator
        if isinstance(calibrator, MCal_Test):
            # MCal_Test uses target probabilities and kappa
            stats = calibrator.fit(
                ablated_probs=ablated_probs_train,
                target_probs=clean_probs_train,
                kappa=1.0,  # Fixed kappa as in the notebook
                max_steps=5000,
                lr=0.01,
                verbose=False
            )
        elif isinstance(calibrator, MCal_CE):
            # MCal_CE uses target labels
            clean_labels_train = clean_probs_train.argmax(dim=1)
            stats = calibrator.fit(
                ablated_probs=ablated_probs_train,
                target_labels=clean_labels_train,
                max_steps=5000,
                lr=0.001,
                verbose=False
            )
        
        # Evaluate on test set
        with torch.no_grad():
            if isinstance(calibrator, MCal_CE):
                calibrated_probs_test = calibrator(ablated_probs_test, return_logits=False)
            else:
                calibrated_probs_test = calibrator(ablated_probs_test)
            
            # Move to CPU for metrics
            calibrated_probs_test = calibrated_probs_test.cpu()
            clean_probs_test_cpu = clean_probs_test.cpu()
            ablated_probs_test_cpu = ablated_probs_test.cpu()
            y_test_cpu = y_test.cpu()
            
            # Calculate metrics
            calibrated_preds = calibrated_probs_test.argmax(dim=1)
            clean_preds = clean_probs_test_cpu.argmax(dim=1)
            
            # Accuracy against clean predictions
            accuracy_vs_clean = (calibrated_preds == clean_preds).float().mean().item()
            
            # Accuracy against true labels
            accuracy_vs_true = (calibrated_preds == y_test_cpu).float().mean().item()
            
            # F1 score against true labels
            f1 = f1_score(y_test_cpu.numpy(), calibrated_preds.numpy(), 
                         average='macro', zero_division=0)
            
            # Missingness bias
            bias = missingness_bias(calibrated_probs_test, clean_probs_test_cpu)
            bias_value = bias.item() if hasattr(bias, 'item') else bias
            
            # Store results
            results[name] = {
                'accuracy_vs_clean': accuracy_vs_clean,
                'accuracy_vs_true': accuracy_vs_true,
                'f1_score': f1,
                'missingness_bias': bias_value,
                'final_loss': stats['loss'][-1],
                'final_train_acc': stats['acc'][-1]
            }
            
            print(f" Done! (Acc vs clean: {accuracy_vs_clean:.4f}, Acc vs true: {accuracy_vs_true:.4f}, Bias: {bias_value:.4f})")
    
    return results, ablated_probs_test_cpu, clean_probs_test_cpu


def print_results(results: Dict[str, Any], ablated_probs_test: torch.Tensor, 
                 clean_probs_test: torch.Tensor, y_test: torch.Tensor) -> None:
    """Print evaluation results in a formatted table."""
    
    # Calculate baseline metrics
    ablated_preds = ablated_probs_test.argmax(dim=1)
    clean_preds = clean_probs_test.argmax(dim=1)
    baseline_acc_vs_clean = (ablated_preds == clean_preds).float().mean().item()
    baseline_acc_vs_true = (ablated_preds == y_test).float().mean().item()
    baseline_f1 = f1_score(y_test.numpy(), ablated_preds.numpy(), 
                           average='macro', zero_division=0)
    baseline_bias = missingness_bias(ablated_probs_test, clean_probs_test)
    baseline_bias_value = baseline_bias.item() if hasattr(baseline_bias, 'item') else baseline_bias
    
    print("\n" + "="*110)
    print(f"{'Model'.ljust(25)}{'Acc(Clean)'.ljust(12)}{'Acc(True)'.ljust(12)}{'F1 Score'.ljust(12)}{'Bias (KL)'.ljust(12)}{'Final Loss'.ljust(12)}{'Train Acc'}")
    print("-"*110)
    
    # Print baseline
    print(f"{'Baseline (No Calib)'.ljust(25)}{baseline_acc_vs_clean:.4f}".ljust(37) + 
          f"{baseline_acc_vs_true:.4f}".ljust(12) + 
          f"{baseline_f1:.4f}".ljust(12) + f"{baseline_bias_value:.4f}".ljust(12) + 
          f"N/A".ljust(12) + "N/A")
    
    # Print calibrator results
    for name, metrics in results.items():
        print(f"{name.ljust(25)}{metrics['accuracy_vs_clean']:.4f}".ljust(37) + 
              f"{metrics['accuracy_vs_true']:.4f}".ljust(12) + 
              f"{metrics['f1_score']:.4f}".ljust(12) + 
              f"{metrics['missingness_bias']:.4f}".ljust(12) + 
              f"{metrics['final_loss']:.4f}".ljust(12) + 
              f"{metrics['final_train_acc']:.4f}")
    
    print("="*110)


def main():
    """Run the main experiment."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate dataset
    print("Generating imbalanced dataset...")
    X_data, y_data = create_imbalanced_dataset(num_samples=2000)
    
    # Split into train and test
    train_size = 1500
    X_train, X_test = X_data[:train_size], X_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Class distribution (train): {torch.bincount(y_train)}")
    print(f"Class distribution (test): {torch.bincount(y_test)}")
    
    # Train base model
    print("\nTraining base model...")
    base_model = BaseModel()
    train_base_model(base_model, X_train, y_train, epochs=500)
    base_model.eval()
    
    # Train and evaluate calibrators
    results, ablated_probs_test, clean_probs_test = train_and_evaluate_calibrators(
        base_model, X_train, y_train, X_test, y_test
    )
    
    # Print results
    print_results(results, ablated_probs_test, clean_probs_test, y_test)
    
    # Print summary
    print("\nSummary:")
    print("-"*50)
    best_acc_clean = max(results.items(), key=lambda x: x[1]['accuracy_vs_clean'])
    best_acc_true = max(results.items(), key=lambda x: x[1]['accuracy_vs_true'])
    best_bias = min(results.items(), key=lambda x: x[1]['missingness_bias'])
    print(f"Best accuracy vs clean: {best_acc_clean[0]} ({best_acc_clean[1]['accuracy_vs_clean']:.4f})")
    print(f"Best accuracy vs true: {best_acc_true[0]} ({best_acc_true[1]['accuracy_vs_true']:.4f})")
    print(f"Best bias reduction: {best_bias[0]} ({best_bias[1]['missingness_bias']:.4f})")


if __name__ == '__main__':
    main()