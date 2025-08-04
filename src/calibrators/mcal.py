"""MCal - Vector scaling calibration model."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseCalibrator

# Import kl_divergence from utils
import sys
import os
from pathlib import Path

# Add the utils directory to path for direct import
current_dir = Path(__file__).parent
utils_dir = current_dir.parent / "utils"
sys.path.insert(0, str(utils_dir))

try:
    from optimization import kl_divergence
except ImportError:
    # Fallback for direct execution
    from utils.optimization import kl_divergence


class MCal(BaseCalibrator):
    """Vector scaling calibration model for probability distributions.
    
    This model implements a learnable vector scaling approach to calibrate
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
        target_distribution: Optional[torch.Tensor] = None,
        ablated_probs: Optional[torch.Tensor] = None,
        clean_probs: Optional[torch.Tensor] = None,
    ):
        super().__init__(num_classes)
        # Initialize parameters exactly like LogitsSharp (lambda_vars and beta_vars both start as ones)
        self.w = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.ones(num_classes))
        
        # Set target distribution (default to uniform)
        if target_distribution is not None:
            self.register_buffer('target_distribution', target_distribution)
        else:
            uniform_dist = torch.ones(num_classes) / num_classes
            self.register_buffer('target_distribution', uniform_dist)

        if ablated_probs is not None:
            # For backward compatibility, support clean_probs but prefer target_distribution
            if clean_probs is not None and target_distribution is None:
                # Legacy mode: derive target from clean_probs expectation
                clean_expectation = clean_probs.mean(dim=0)
                self.register_buffer('target_distribution', clean_expectation)
            self.fit(ablated_probs)

    def fit(
        self,
        ablated_probs: torch.Tensor,
        target_distribution: Optional[torch.Tensor] = None,
        kappa: float = 1.0,
        max_steps: int = 10000,
        lr: float = 1e-1,
        early_stopping: bool = False,
        ema_decay: float = 0.9,
        scale_before_sharpen: bool = True,
        verbose: bool = False,
        use_random_targets: bool = False,
        # Backward compatibility
        clean_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Fit the calibration model to the given probability distributions.
        
        Args:
            ablated_probs (torch.Tensor): Ablated probability distributions
            target_distribution (Optional[torch.Tensor]): Target distribution to optimize towards (defaults to uniform)
            kappa (float): Sharpening factor for probability distributions
            max_steps (int): Maximum number of optimization steps
            lr (float): Learning rate for optimization
            early_stopping (bool): Whether to use early stopping based on gradient norm
            ema_decay (float): Exponential moving average decay for gradient norm tracking
            scale_before_sharpen (bool): Whether to scale before sharpening
            verbose (bool): Whether to show progress bar and metrics
            use_random_targets (bool): Whether to use random targets (for debugging)
            clean_probs (Optional[torch.Tensor]): Legacy parameter for backward compatibility
            
        Returns:
            Dictionary containing training statistics
        """
        self._validate_fit_inputs(ablated_probs, target_distribution)
        
        # Set target distribution for this fit call
        if target_distribution is not None:
            target_dist = target_distribution.to(ablated_probs.device)
        elif clean_probs is not None:
            # Legacy backward compatibility: use clean_probs expectation as target
            target_dist = clean_probs.mean(dim=0)
            if verbose:
                print("Using clean_probs expectation as target (legacy mode)")
        else:
            # Use the default target distribution (uniform)
            target_dist = self.target_distribution.to(ablated_probs.device)
        
        if use_random_targets:
            # Use random targets for debugging (as in original implementation)
            target_classes = torch.eye(self.num_classes)[
                torch.randperm(ablated_probs.shape[0]) % self.num_classes
            ]
        else:
            # Use target distribution repeated for each sample
            target_classes = target_dist.unsqueeze(0).expand(ablated_probs.shape[0], -1)

        # Initialize gradient norm tracking
        grad_norm_avg = 0.0  # The moving average of the gradient norm
        grad_norm_ref = 0.0  # The reference gradient norm
        warmup_steps = 10    # The number of steps to warmup the gradient norm

        stats = {
            "loss": [],
            "acc": [],
            "grad_norm": [],
        }

        optimizer = optim.Adam(self.parameters(), lr=lr)

        pbar = tqdm(range(max_steps), desc="MCal Training") if verbose else range(max_steps)
        for step in pbar:
            optimizer.zero_grad()

            # Forward pass and sharpening
            q = self.forward(ablated_probs)
            
            # Check for NaN/inf in forward pass
            if torch.isnan(q).any() or torch.isinf(q).any():
                if verbose:
                    print(f"NaN/Inf detected in forward pass at step {step}")
                    print(f"w: {self.w.data}")
                    print(f"b: {self.b.data}")
                    print(f"q stats: min={q.min():.6f}, max={q.max():.6f}, mean={q.mean():.6f}")
                break
                
            s = (q / q.max(dim=1, keepdim=True).values) if scale_before_sharpen else q

            # Apply sharpening with numerical stability
            s = torch.clamp(s, min=1e-8, max=1.0)  # Clamp to avoid numerical issues
            s = s ** kappa
            s = s / s.sum(dim=1, keepdim=True)
            
            # Check for NaN/inf after sharpening
            if torch.isnan(s).any() or torch.isinf(s).any():
                if verbose:
                    print(f"NaN/Inf detected in sharpening at step {step}")
                    print(f"s stats: min={s.min():.6f}, max={s.max():.6f}, mean={s.mean():.6f}")
                break

            # Compute loss against target distribution (expectation-based like LogitsSharp)
            s_expectation = s.mean(dim=0)  # Get expectation across samples
            loss = kl_divergence(s_expectation, target_dist)
            
            # Check for NaN/inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"NaN/Inf detected in loss at step {step}: {loss.item()}")
                break
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Compute metrics - accuracy against target distribution argmax
            target_argmax = target_dist.argmax() if target_dist.dim() == 1 else target_classes.argmax(dim=1)
            acc = (s.argmax(dim=1) == target_argmax).float().mean()
            grad_norm = self.w.grad.norm(p="fro") + self.b.grad.norm(p="fro")

            stats["loss"].append(loss.item())
            stats["acc"].append(acc.item())
            stats["grad_norm"].append(grad_norm.item())

            # Early stopping based on gradient norm (commented out as requested)
            # if step < warmup_steps:
            #     grad_norm_avg += grad_norm / warmup_steps
            #     grad_norm_ref = grad_norm_avg
            # else:
            #     grad_norm_avg = ema_decay * grad_norm_avg + (1 - ema_decay) * grad_norm
            #     
            #     if early_stopping and grad_norm_avg < max(0.01 * grad_norm_ref, 1e-6):
            #         if verbose:
            #             print(f"Early stopping at step {step}")
            #         break

            if verbose:
                pbar.set_description(f"Loss: {loss.item():.3e}, Acc: {acc:.3f}, GradNorm: {grad_norm:.3e}")
                
            # Additional debugging every 100 steps
            if verbose and step % 100 == 0:
                print(f"Step {step}: w={self.w.data.mean():.6f}±{self.w.data.std():.6f}, "
                      f"b={self.b.data.mean():.6f}±{self.b.data.std():.6f}")

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
        
        # Convert probabilities to log space for numerical stability (match LogitsSharp bounds)
        probs_clamped = torch.clamp(probs, min=1e-6, max=1 - 1e-6)
        z = torch.log(probs_clamped)
        
        # Apply learned scaling and bias (no clamping to match LogitsSharp exactly)
        logits = self.w.view(1, -1) * z + self.b.view(1, -1)
        
        # Apply temperature scaling to prevent overflow in softmax
        temperature = 1.0
        q = F.softmax(logits / temperature, dim=1)
        
        return q