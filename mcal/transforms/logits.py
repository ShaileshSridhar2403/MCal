"""Logit-based transformation methods."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Any, Optional

from .base import BaseTransform

from ..utils.optimization import get_expectation, make_one_hot, kl_divergence


class LogitsSharpTransform(BaseTransform):
    """Logit sharpening transformation with learned parameters."""
    
    def __init__(self, device: Optional[torch.device] = None, name: Optional[str] = None):
        super().__init__(device, name if name is not None else "logits_sharp")
        self.lambdas = None
        self.betas = None
        self.kappa = 1.0  # Sharpening factor
    
    def fit(self, tensor_path: str, batch_size: int = 256, num_epochs: int = 1000, **kwargs) -> Dict[str, Any]:
        """Fit logit sharpening parameters.
        
        Args:
            tensor_path: Path to tensor file with training data
            batch_size: Batch size for optimization (unused in this implementation) 
            num_epochs: Number of optimization epochs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with training statistics
        """
        full_tensor = np.load(tensor_path)
        n_fractions, n_samples, n_outputs = full_tensor.shape
        
        # Initialize parameters
        self.lambdas = np.zeros((n_fractions, n_outputs))
        self.betas = np.zeros((n_fractions, n_outputs))
        
        stats = {"losses": [], "convergence_epochs": []}
        
        # Optimize parameters for each fraction
        for fraction in tqdm(range(n_fractions), desc="Computing logits sharp parameters"):
            predictions = torch.tensor(full_tensor[fraction], dtype=torch.float32).to(self.device)
            
            # Optimize lambda and beta parameters
            result = self._find_optimal_lambda_batch_logits(
                predictions, 
                num_epochs=num_epochs
            )
            
            lambdas, betas, loss, _, _, _ = result
            self.lambdas[fraction] = lambdas.cpu().numpy()
            self.betas[fraction] = betas.cpu().numpy()
            
            stats["losses"].append(loss)
        
        self._is_fitted = True
        return stats
    
    def _find_optimal_lambda_batch_logits(self, predictions, num_epochs=1000):
        """Find optimal lambda and beta parameters for logit sharpening."""
        probs = predictions.to(self.device)
        N, dim = probs.shape
        best_loss = 10000
        best_result = None
        
        # Initialize parameters
        lambda_vars = torch.ones(dim, requires_grad=True, device=self.device)
        beta_vars = torch.ones(dim, requires_grad=True, device=self.device)
        
        optimizer = optim.Adam([lambda_vars, beta_vars], lr=1e-1)
        loss_history = []
        
        pbar = tqdm(range(num_epochs), desc="Optimizing parameters", leave=False)
        for epoch in pbar:
            # Convert probabilities to logits
            z = torch.log(probs.clamp(min=1e-6, max=(1 - 1e-6)))
            
            # Apply transformation
            q = F.softmax(lambda_vars * z + beta_vars, dim=-1)
            
            # Apply sharpening with kappa = 1 (as in XAI_Benchmark)
            if self.kappa != 1.0:
                t = q / q.max(dim=-1, keepdim=True).values
                q_sharp = (t ** self.kappa) / (t ** self.kappa).sum(dim=-1, keepdim=True)
            else:
                q_sharp = q
            
            # Calculate KL divergence loss
            uniform_dist = torch.ones(dim, device=self.device) / dim
            loss = kl_divergence(q_sharp.mean(dim=0), uniform_dist)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = lambda_vars.grad.norm(p=2).item()
            optimizer.step()
            
            pbar.set_description(f"loss {loss:.4e}, gnorm {grad_norm:.4e}")
            loss_history.append(loss.item())
            
            if loss < best_loss:
                best_loss = loss
                best_result = (
                    lambda_vars.clone().detach(),
                    beta_vars.clone().detach(),
                    loss.item(),
                    make_one_hot(q_sharp).mean(dim=0).detach(),
                    q_sharp.detach(),
                    loss.item()
                )
        
        return best_result
    
    def transform(self, input_tensor: torch.Tensor, fraction_idx: int = 0) -> torch.Tensor:
        """Apply logit sharpening transformation.
        
        Args:
            input_tensor: Input probability tensor
            fraction_idx: Which fraction's parameters to use (default: 0)
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")
        
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        
        input_tensor = input_tensor.to(self.device)
        
        # Use parameters from specified fraction
        fraction_idx = min(fraction_idx, self.lambdas.shape[0] - 1)  # Ensure valid index
        lambdas = torch.tensor(self.lambdas[fraction_idx], device=self.device)
        betas = torch.tensor(self.betas[fraction_idx], device=self.device)
        
        # Convert probabilities to logits
        z = torch.log(input_tensor.clamp(min=1e-6, max=1-1e-6))
        
        # Apply learned transformation
        q = F.softmax(z * lambdas + betas, dim=-1)
        
        # Apply sharpening (kappa=1 means no additional sharpening)
        if self.kappa != 1.0:
            t = q / q.max(dim=-1, keepdim=True).values
            q_sharp = (t ** self.kappa) / (t ** self.kappa).sum(dim=-1, keepdim=True)
            return q_sharp
        else:
            return q
    
    def save(self, path: str) -> None:
        """Save transformation parameters."""
        save_dict = {
            'lambdas': self.lambdas,
            'betas': self.betas,
            'kappa': self.kappa
        }
        np.save(path, save_dict)
    
    def load(self, path: str) -> None:
        """Load transformation parameters."""
        save_dict = np.load(path, allow_pickle=True).item()
        self.lambdas = save_dict['lambdas']
        self.betas = save_dict['betas']
        self.kappa = save_dict.get('kappa', 1.0)
        self._is_fitted = True

