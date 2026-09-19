"""Base transform interface."""

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTransform(ABC):
    """Base class for all transformation methods.
    
    This class defines the interface that all transformation classes should implement.
    Transformations are used to modify probability distributions for calibration purposes.
    """
    
    def __init__(self, device: Optional[torch.device] = None, name: Optional[str] = None):
        """Initialize the transform.
        
        Args:
            device: Device to run computations on. If None, uses CUDA if available, else CPU.
            name: Name of the transform. If None, uses class name.
        """
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.name = name if name is not None else self.__class__.__name__
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, tensor_path: str, **kwargs) -> Dict[str, Any]:
        """Train/optimize the transformation.
        
        Args:
            tensor_path: Path to tensor file containing training data
            **kwargs: Additional arguments specific to the transform
            
        Returns:
            Dictionary containing training statistics or metadata
        """
        pass
    
    @abstractmethod
    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to input tensor.
        
        Args:
            input_tensor: Input probability distributions to transform
            
        Returns:
            Transformed probability distributions
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the transformation parameters.
        
        Args:
            path: Path to save the transformation parameters
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the transformation parameters.
        
        Args:
            path: Path to load the transformation parameters from
        """
        pass
    
    @property
    def is_fitted(self) -> bool:
        """Check if the transform has been fitted."""
        return self._is_fitted
    
    def __repr__(self) -> str:
        """String representation of the transform."""
        return f"{self.name}(device={self.device}, fitted={self.is_fitted})"