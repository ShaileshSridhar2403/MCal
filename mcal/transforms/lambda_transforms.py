"""Lambda-based transformation methods."""

import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any, Optional

from .base import BaseTransform

from ..utils.optimization import get_expectation, find_optimal_lambda_batch, apply_lambda_adjustment


class LambdaTransform(BaseTransform):
    """Base class for lambda-based transformations."""
    
    def __init__(self, device: Optional[torch.device] = None, name: Optional[str] = None):
        super().__init__(device, name)
        self.lambdas = None
    
    def save(self, path: str) -> None:
        """Save lambda parameters to file."""
        if self.lambdas is None:
            raise ValueError("No lambda parameters to save. Fit the transform first.")
        np.save(path, self.lambdas)
    
    def load(self, path: str) -> None:
        """Load lambda parameters from file."""
        self.lambdas = np.load(path)
        self._is_fitted = True


class OptimizedLambdaTransform(LambdaTransform):
    """Transform using optimized lambda values via KL divergence minimization."""
    
    def __init__(self, device: Optional[torch.device] = None, name: Optional[str] = None):
        super().__init__(device, name if name is not None else "optimized_lambda")
    
    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply optimized lambda transformation."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")
        
        return apply_lambda_adjustment(
            input_tensor, 
            self.lambdas, 
            self.device, 
            normalization=True
        )
    
    def fit(
        self, 
        tensor_path: str, 
        batch_size: int = 256, 
        num_epochs: int = 1000, 
        **kwargs
    ) -> Dict[str, Any]:
        """Fit optimized lambda parameters.
        
        Args:
            tensor_path: Path to tensor file with shape (n_fractions, n_samples, n_outputs)
            batch_size: Batch size for optimization
            num_epochs: Number of optimization epochs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with fitting statistics
        """
        full_tensor = np.load(tensor_path)
        n_fractions, n_samples, n_outputs = full_tensor.shape
        
        self.lambdas = np.zeros((n_fractions, n_outputs))
        stats = {"losses": [], "convergence_epochs": []}
        
        for fraction in tqdm(range(n_fractions), desc="Computing optimized lambdas"):
            predictions = torch.tensor(full_tensor[fraction], dtype=torch.float32).to(self.device)
            
            result = find_optimal_lambda_batch(
                predictions,
                self.device,
                batch_size=batch_size,
                num_epochs=num_epochs
            )
            
            lambdas, loss, _, _, final_loss = result
            self.lambdas[fraction] = lambdas.cpu().numpy()
            
            stats["losses"].append(final_loss)
        
        self._is_fitted = True
        return stats


class ExpectationLambdaTransform(LambdaTransform):
    """Transform using inverse of expectation distributions."""
    
    def __init__(
        self, 
        device: Optional[torch.device] = None, 
        method: str = 'prob', 
        name: Optional[str] = None
    ):
        """Initialize expectation-based lambda transform.
        
        Args:
            device: Device to run computations on
            method: Either 'prob' or 'onehot' for probability or one-hot based expectations
            name: Optional name for the transform
        """
        if method not in ['prob', 'onehot']:
            raise ValueError("method must be either 'prob' or 'onehot'")
        
        name_suffix = f"expectation_{method}"
        super().__init__(device, name if name is not None else name_suffix)
        self.method = method
    
    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply expectation-based lambda transformation."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")
        
        # Apply normalization only for one-hot method (as per original code)
        normalization = False
        return apply_lambda_adjustment(
            input_tensor, 
            self.lambdas, 
            self.device, 
            normalization=normalization
        )
    
    def fit(self, tensor_path: str, **kwargs) -> Dict[str, Any]:
        """Fit expectation-based lambda parameters.
        
        Args:
            tensor_path: Path to tensor file with shape (n_fractions, n_samples, n_outputs)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with fitting statistics
        """
        full_tensor = np.load(tensor_path)
        n_fractions, n_samples, n_outputs = full_tensor.shape
        
        self.lambdas = np.zeros((n_fractions, n_outputs))
        stats = {"method": self.method, "expectations": []}
        
        method_name = "probability" if self.method == "prob" else "one-hot"
        
        for fraction in tqdm(range(n_fractions), desc=f"Computing {method_name}-based lambdas"):
            predictions = torch.tensor(full_tensor[fraction], dtype=torch.float32).to(self.device)

            # Get expectations
            one_hot_expectation, prob_expectation = get_expectation(
                predictions, 
                self.device, 
                normalization=False
            )
            
            if self.method == 'prob':
                expectation = prob_expectation
            else:  # onehot
                expectation = one_hot_expectation
            
            # Compute inverse expectation as lambdas
            lambdas = 1 / expectation
            self.lambdas[fraction] = lambdas.cpu().numpy()
            
            stats["expectations"].append(expectation.cpu().numpy())
        
        self._is_fitted = True
        return stats

