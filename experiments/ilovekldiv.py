import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from typing import Tuple, Dict

from experiment_utils import kl_divergence, missingness_bias


# For export to other modules (referenced in mcal_test.py)
calibrated_logits_hybrid = None


def create_distractor_dataset(num_samples: int = 2000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a dataset where CE loss fails on both accuracy and bias.
    
    Class 0: "Strong A" [1, 0, 1, 0, noise...] (Rare - 10%)
    Class 1: "Strong B" [0, 1, 0, 1, noise...] (Rare - 10%)
    Class 2: "Distractor C" [0, 0, 0, 0, noise...] (Common - 80%)
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        X: Features tensor of shape (num_samples, 6)
        y: Labels tensor of shape (num_samples,)
    """
    # Define class distribution
    class_probs = torch.tensor([0.1, 0.1, 0.80])
    y = torch.multinomial(class_probs, num_samples, replacement=True)
    
    # Initialize features
    X = torch.zeros(num_samples, 6, dtype=torch.float32)
    
    # Set pattern for each class
    mask_a = (y == 0)
    X[mask_a, 0] = 1.0
    X[mask_a, 2] = 1.0  # Class 0: Strong A pattern
    
    mask_b = (y == 1)
    X[mask_b, 1] = 1.0
    X[mask_b, 3] = 1.0  # Class 1: Strong B pattern
    # Class 2 remains all zeros for first 4 features
    
    # Add noise to last 2 features
    X[:, 4:] = torch.rand(num_samples, 2) * 0.1
    
    return X, y


class BaseModel(nn.Module):
    """Simple feedforward neural network for classification."""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Calibrator(nn.Module):
    """Calibration model that adjusts logits from a base model."""
    
    def __init__(self, num_classes: int = 3, mode: str = "linear"):
        super().__init__()
        self.mode = mode
        
        if mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(num_classes, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        elif mode == "linear":
            self.net = nn.Linear(num_classes, num_classes)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'mlp' or 'linear'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                epochs: int = 500, lr: float = 0.01) -> None:
    """Train a model using cross-entropy loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()


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

def train_calibrators(base_model: nn.Module, X_train: torch.Tensor, 
                     calibrator_mode: str = "linear", num_epochs: int = 500) -> Dict[str, nn.Module]:
    """
    Train different calibration models with various loss functions.
    
    Returns:
        Dictionary containing trained calibrators
    """
    # Initialize calibrators
    calibrators = {
        'ce': Calibrator(mode=calibrator_mode),
        'kl': Calibrator(mode=calibrator_mode),
    }
    
    # Add hybrid calibrators
    lambda_values = [0.2, 0.4, 0.6, 0.8]
    for lam in lambda_values:
        calibrators[f'hybrid_{lam}'] = Calibrator(mode=calibrator_mode)
    
    # Initialize optimizers
    optimizers = {name: optim.Adam(cal.parameters(), lr=0.01) 
                  for name, cal in calibrators.items()}
    
    print(f"\nTraining {calibrator_mode} calibrators for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Generate ablated data
        X_ablated = ablate_data(X_train)
        
        with torch.no_grad():
            clean_logits = base_model(X_train)
            ablated_logits = base_model(X_ablated)
            clean_freq = F.one_hot(clean_logits.argmax(dim=1), num_classes=3).float().mean(dim=0)
        
        # Train CE calibrator (instance-level)
        optimizers['ce'].zero_grad()
        ce_calibrated_logits = calibrators['ce'](ablated_logits)
        loss_ce = F.cross_entropy(ce_calibrated_logits, clean_logits.argmax(dim=1))
        loss_ce.backward()
        optimizers['ce'].step()
        
        # Train KL calibrator (aggregate-level)
        optimizers['kl'].zero_grad()
        kl_calibrated_logits = calibrators['kl'](ablated_logits)
        kl_calibrated_freq = F.softmax(kl_calibrated_logits, dim=1).mean(dim=0)
        # Ensure we get a tensor for backprop
        loss_kl = kl_divergence(kl_calibrated_freq.unsqueeze(0), clean_freq.unsqueeze(0)).squeeze()
        loss_kl.backward()
        optimizers['kl'].step()
        
        # Train hybrid calibrators
        for lam in lambda_values:
            name = f'hybrid_{lam}'
            optimizers[name].zero_grad()
            
            hybrid_calibrated_logits = calibrators[name](ablated_logits)
            hybrid_calibrated_freq = F.softmax(hybrid_calibrated_logits, dim=1).mean(dim=0)
            
            # Combine losses
            loss_kl_hybrid = kl_divergence(hybrid_calibrated_freq.unsqueeze(0), clean_freq.unsqueeze(0)).squeeze()
            loss_ce_hybrid = F.cross_entropy(hybrid_calibrated_logits, clean_logits.argmax(dim=1))
            loss_hybrid = (1 - lam) * loss_ce_hybrid + lam * loss_kl_hybrid
            
            loss_hybrid.backward()
            optimizers[name].step()
    
    print("Calibrators trained.")
    return calibrators


def evaluate_calibrators(base_model: nn.Module, calibrators: Dict[str, nn.Module],
                        X_test: torch.Tensor, y_test: torch.Tensor) -> None:
    """Evaluate all calibrators on test data."""
    # Set models to evaluation mode
    base_model.eval()
    for cal in calibrators.values():
        cal.eval()
    
    # Generate ablated test data
    X_test_ablated = ablate_data(X_test, ablation_prob=0.5)
    
    with torch.no_grad():
        # Get base model predictions
        base_logits = base_model(X_test_ablated)
        base_probs = F.softmax(base_logits, dim=1)
        
        # Get clean predictions for bias calculation
        clean_probs = F.softmax(base_model(X_test), dim=1)
        
        # Evaluation helper
        def compute_metrics(probs: torch.Tensor) -> Tuple[float, float, float]:
            """Compute accuracy, F1 score, and missingness bias."""
            preds = probs.argmax(dim=1)
            acc = (preds == y_test).float().mean().item()
            f1 = f1_score(y_test.cpu(), preds.cpu(), average='macro', zero_division=0)
            bias_value = missingness_bias(probs, clean_probs)
            # Handle both tensor and float returns
            bias = bias_value.item() if hasattr(bias_value, 'item') else bias_value
            return acc, f1, bias
        
        # Print header
        print("\n" + "="*70)
        print(f"{'Model'.ljust(30)}{'Accuracy'.ljust(12)}{'F1 Score'.ljust(12)}{'Missingness Bias (KL)'}")
        print("-"*70)
        
        # Evaluate base model
        acc, f1, bias = compute_metrics(base_probs)
        print(f"{'Base Model'.ljust(30)}{acc:.2%}".ljust(42) + f"{f1:.2%}".ljust(12) + f"{bias:.4f}")
        
        # Evaluate calibrators
        for name, calibrator in calibrators.items():
            calibrated_logits = calibrator(base_logits)
            calibrated_probs = F.softmax(calibrated_logits, dim=1)
            
            acc, f1, bias = compute_metrics(calibrated_probs)
            
            display_name = name.replace('_', ' λ=') if 'hybrid' in name else f"{name.upper()} Calibrator"
            print(f"{display_name.ljust(30)}{acc:.2%}".ljust(42) + f"{f1:.2%}".ljust(12) + f"{bias:.4f}")
        
        print("="*70)


def main():
    """Run the main experiment."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate dataset
    X_data, y_data = create_distractor_dataset()
    X_train, X_test = X_data[:1500], X_data[1500:]
    y_train, y_test = y_data[:1500], y_data[1500:]
    
    # Train base model
    print("Training Base Model...")
    base_model = BaseModel()
    train_model(base_model, X_train, y_train, epochs=500)
    base_model.eval()
    
    # Run experiments for both calibrator types
    for mode in ["linear", "mlp"]:
        print(f"\n{'='*50}")
        print(f"Experiment with {mode.upper()} calibrators")
        print(f"{'='*50}")
        
        # Train calibrators
        calibrators = train_calibrators(base_model, X_train, 
                                      calibrator_mode=mode, num_epochs=500)
        
        # Evaluate
        print(f"\nEvaluation Results ({mode.upper()} calibrators):")
        evaluate_calibrators(base_model, calibrators, X_test, y_test)


if __name__ == '__main__':
    main()
