"""Platt scaling calibration model."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseCalibrator


class PlattCalibrator(BaseCalibrator):
    """Platt scaling calibration model for probability distributions.
    
    This model implements a learnable scaling approach to calibrate
    probability distributions. It learns class-specific scaling parameters (w) and
    bias terms (b) to adjust the input probabilities.
    
    Args:
        num_classes (int): Number of classes in the classification task
        ablated_probs (Optional[torch.Tensor]): Initial ablated probabilities for fitting
        clean_probs (Optional[torch.Tensor]): Initial clean probabilities for fitting
    """
    
    def __init__(
        self,
        num_classes: int,
        ablated_probs: Optional[torch.Tensor] = None,
        clean_probs: Optional[torch.Tensor] = None,
    ):
        super().__init__(num_classes)
        self.w = nn.Parameter(torch.ones(num_classes))   # Class-specific scaling parameters
        self.b = nn.Parameter(torch.zeros(num_classes))  # Class-specific bias terms
        
        if ablated_probs is not None and clean_probs is not None:
            self.fit(ablated_probs, clean_probs)

    def fit(
        self,
        ablated_probs: torch.Tensor,
        clean_probs: torch.Tensor,
        max_steps: int = 10000,
        lr: float = 1e-2,
        verbose: bool = False,
        use_random_targets: bool = True,
    ) -> Dict[str, Any]:
        """Fit the calibration model to the given probability distributions.

        Args:
            ablated_probs (torch.Tensor): Ablated probability distributions
            clean_probs (torch.Tensor): Clean probability distributions
            max_steps (int): Maximum number of optimization steps
            lr (float): Learning rate for optimization
            verbose (bool): Whether to show progress bar and metrics
            use_random_targets (bool): Whether to use random targets (for debugging)
            
        Returns:
            Dictionary containing training statistics
        """
        self._validate_fit_inputs(ablated_probs, clean_probs)
        
        if use_random_targets:
            # Use random targets for debugging (as in original implementation)
            clean_classes = torch.eye(self.num_classes)[
                torch.randperm(clean_probs.shape[0]) % self.num_classes
            ]
        else:
            # Use actual one-hot encoding of clean predictions
            clean_classes = F.one_hot(clean_probs.argmax(dim=1), num_classes=self.num_classes).float()
            
        optimizer = optim.Adam(self.parameters(), lr=lr)
        nll_criterion = nn.CrossEntropyLoss()

        stats = {
            "loss": [],
            "acc": [],
        }

        pbar = tqdm(range(max_steps), desc="Platt Training") if verbose else range(max_steps)
        for step in pbar:
            optimizer.zero_grad()

            q = self.forward(ablated_probs)
            loss = nll_criterion(q, clean_probs)
            loss.backward()
            optimizer.step()

            acc = (q.argmax(dim=1) == clean_probs.argmax(dim=1)).float().mean()

            stats["loss"].append(loss.item())
            stats["acc"].append(acc.item())

            if verbose:
                pbar.set_description(f"Loss: {loss.item():.3e}, Acc: {acc:.3f}")

        self._is_fitted = True
        return stats
            
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the calibration model.
        
        Args:
            probs (torch.Tensor): Input probability distributions of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: Calibrated probability distributions
        """
        self._validate_input_probs(probs)
        
        # Convert probabilities to log space for numerical stability
        z = torch.log(probs.clamp(1e-6, 1 - 1e-6))
        
        # Apply learned scaling and bias
        q = F.softmax(self.w.view(1, -1) * z + self.b.view(1, -1), dim=1)
        return q