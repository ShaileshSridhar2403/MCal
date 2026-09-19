"""Base calibrator interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseCalibrator(nn.Module, ABC):
    """Base class for probability calibrators.
    
    All calibrators should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self, 
        ablated_probs: torch.Tensor, 
        **kwargs
    ) -> Dict[str, Any]:
        """Fit the calibrator to the given probability distributions.
        
        Args:
            ablated_probs: Ablated probability distributions of shape (batch_size, num_classes)
            **kwargs: Additional arguments specific to the calibrator
            
        Returns:
            Dictionary containing training statistics
        """
    
    @abstractmethod
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply calibration to input probabilities.
        
        Args:
            probs: Input probability distributions of shape (batch_size, num_classes)
            
        Returns:
            Calibrated probability distributions of same shape
        """
    
    def predict(self, probs: torch.Tensor) -> torch.Tensor:
        """Convenience method for forward pass.
        
        Args:
            probs: Input probability distributions
            
        Returns:
            Calibrated probability distributions
        """
        return self.forward(probs)
    
    @property
    def is_fitted(self) -> bool:
        """Check if the calibrator has been fitted."""
        return self._is_fitted
    
    def _validate_input_probs(self, probs: torch.Tensor) -> None:
        """Validate input probability tensor.
        
        Args:
            probs: Probability tensor to validate
            
        Raises:
            AssertionError: If probabilities are not valid
        """
        assert isinstance(probs, torch.Tensor), "Input must be a torch.Tensor"
        assert probs.dim() == 2, f"Expected 2D tensor, got {probs.dim()}D"
        assert probs.shape[1] == self.num_classes, f"Expected {self.num_classes} classes, got {probs.shape[1]}"
        assert torch.all(probs >= -1e-8) and torch.all(probs <= 1 + 1e-8), "Expected probability values between 0 and 1"
    
    def _validate_fit_inputs(self, ablated_probs: torch.Tensor, target_distribution: Optional[torch.Tensor] = None) -> None:
        """Validate inputs for fitting.
        
        Args:
            ablated_probs: Ablated probability distributions
            target_distribution: Optional target distribution
            
        Raises:
            AssertionError: If inputs are not valid
        """
        self._validate_input_probs(ablated_probs)
        if target_distribution is not None:
            assert target_distribution.shape[0] == self.num_classes, f"Target distribution must have {self.num_classes} elements"
            assert torch.allclose(target_distribution.sum(), torch.tensor(1.0), atol=1e-6), "Target distribution must sum to 1"