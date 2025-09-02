"""Temperature scaling calibration model."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseCalibrator


class TemperatureScaling(BaseCalibrator):
    """Temperature scaling calibration model.
    
    This model implements temperature scaling to calibrate probability distributions.
    It learns a single temperature parameter T that scales the logits before applying softmax.
    
    Args:
        num_classes (int): Number of classes in the classification task
    """
    
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def fit(
        self,
        ablated_probs: torch.Tensor,
        labels: torch.Tensor,
        max_steps: int = 1000,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Fit temperature scaling to the given probability distributions.
        
        Args:
            ablated_probs: Ablated probability distributions  
            labels: True class labels (integers)
            max_steps: Maximum number of optimization steps
            lr: Learning rate for optimization
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary containing training statistics
        """
        # self._validate_fit_inputs(ablated_probs, labels)
        
        # Convert probabilities back to logits (approximate)
        ablated_logits = torch.log(ablated_probs.clamp(1e-6, 1-1e-6))
        
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_steps)
        criterion = nn.CrossEntropyLoss()
        
        stats = {
            "loss": [],
            "temperature": [],
        }
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = ablated_logits / self.temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            
            stats["loss"].append(loss.item())
            stats["temperature"].append(self.temperature.item())
            
            return loss
        
        if verbose:
            print("Fitting temperature scaling...")
            
        optimizer.step(closure)
        
        if verbose:
            print(f"Optimal temperature: {self.temperature.item():.4f}")
            
        self._is_fitted = True
        return stats
    
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to input probabilities.
        
        Args:
            probs: Input probability distributions
            
        Returns:
            Temperature-scaled probability distributions
        """
        self._validate_input_probs(probs)
        
        # Convert probabilities back to logits (approximate)
        logits = torch.log(probs.clamp(1e-6, 1-1e-6))
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        calibrated_probs = F.softmax(scaled_logits, dim=1)
        
        return calibrated_probs