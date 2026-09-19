"""MCal_CE - Cross-entropy loss variant of MCal calibration model."""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from .base import BaseCalibrator

# Import kl_divergence from utils
from ..utils.optimization import kl_divergence

logger = logging.getLogger(__name__)



class ResidualBlock(nn.Module):
    """Residual block wrapper for neural network modules."""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class SimpleMCalCE(nn.Module):
    """Simple MCal calibration model using cross-entropy loss."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.head = nn.Linear(num_classes, num_classes)
    
    def forward(self, ablated_logits: torch.Tensor) -> torch.Tensor:
        return self.head(ablated_logits)

    def fit(
        self,
        ablated_logits: torch.Tensor,
        target_labels: torch.Tensor,
        max_steps: int = 5000,
        lr: float = 1e-3,
        verbose: bool = False
    ) -> Dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        stats = {"loss": [], "acc": []}
        pbar = tqdm(range(max_steps), desc="SimpleMCalCE Training") if verbose else range(max_steps)
        for step in pbar:
            optimizer.zero_grad()
            calibrated_logits = self.forward(ablated_logits)
            loss = nn.CrossEntropyLoss()(calibrated_logits, target_labels)
            loss.backward()
            optimizer.step()
            acc = (calibrated_logits.argmax(dim=1) == target_labels).float().mean()
            stats["loss"].append(loss.item())
            stats["acc"].append(acc.item())
            if verbose:
                pbar.set_description(f"Loss: {loss.item():.3e}, Acc: {acc:.3f}")
        return stats


class MCal_CE(BaseCalibrator):
    """MCal calibration model using cross-entropy loss.
    
    This variant of MCal uses cross-entropy loss instead of KL divergence,
    making it more suitable for supervised calibration scenarios.
    
    Args:
        num_classes (int): Number of classes in the classification task
        head_type (str): Type of calibration head:
            - 'linear': Full linear transformation (matrix multiplication)
            - 'scaling': Element-wise scaling with bias (ax + c for each logit)
            - 'mlp': Multi-layer perceptron with residual connection
    """

    def __init__(self, num_classes: int, head_type: str = "linear"):
        super().__init__(num_classes)

        self.head_type = head_type
        if head_type == "linear":
            self.head = nn.Linear(num_classes, num_classes)
        elif head_type == "scaling":
            # Element-wise scaling with bias: ax + c for each logit
            self.scale = nn.Parameter(torch.ones(num_classes))
            self.bias = nn.Parameter(torch.zeros(num_classes))
            self.head = None  # No head for scaling method
        elif head_type == "mlp":
            self.head = ResidualBlock(nn.Sequential(
                nn.Linear(num_classes, 4 * num_classes),
                nn.GELU(),
                nn.Linear(4 * num_classes, num_classes)))
        else:
            raise ValueError(f"Invalid head type: {head_type}. Choose from 'linear', 'scaling', 'mlp'")

    def forward(self, ablated_probs: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass of the calibration model.
        
        Args:
            ablated_probs (torch.Tensor): Ablated probability distributions
            return_logits (bool): Whether to return logits or probabilities
            
        Returns:
            torch.Tensor: Calibrated probability distributions or logits
        """
        self._validate_input_probs(ablated_probs)
        
        ablated_logits = torch.log(ablated_probs.clamp(min=1e-8))
        
        if self.head_type == "scaling":
            # Element-wise scaling with bias: ax + c
            calibrated_logits = self.scale * ablated_logits + self.bias
        else:
            # Use the neural network head (linear or mlp)
            calibrated_logits = self.head(ablated_logits)
            
        if return_logits:
            return calibrated_logits
        else:
            return F.softmax(calibrated_logits, dim=1)

    def fit(
        self,
        ablated_probs: torch.Tensor,
        target_labels: torch.Tensor,
        max_steps: int = 5000,
        lr: float = 1e-3,
        verbose: bool = False,
        fraction: Optional[int] = None,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fit the calibration model using cross-entropy loss.
        
        Args:
            ablated_probs (torch.Tensor): Ablated probability distributions (2D: samples × classes)
            target_labels (torch.Tensor): Target labels extracted from 0th index predictions
            max_steps (int): Maximum number of optimization steps
            lr (float): Learning rate for optimization
            verbose (bool): Whether to show progress bar and metrics
            fraction (int, optional): Ablation fraction this fit belongs to; recorded
                in the fit summary
            experiment_id (str, optional): Experiment identifier recorded in the fit
                summary when ``fraction`` is given (defaults to "default")
            
        Returns:
            Dictionary containing training statistics. A summary of the fitted
            calibrator (mean probabilities, KL divergences, accuracy and learned
            parameters) is stored in ``self.fit_summary_``. ``fit`` writes no files.
        """
        self._validate_fit_inputs(ablated_probs)
        assert target_labels.shape in ((ablated_probs.shape[0],), ablated_probs.shape), \
            "Expected one integer label or one class-probability row per sample"
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        stats = {"loss": [], "acc": []}

        pbar = tqdm(range(max_steps), desc="MCal_CE Training") if verbose else range(max_steps)
        for step in pbar:
            optimizer.zero_grad()
            calibrated_logits = self.forward(ablated_probs, return_logits=True)
            loss = nn.CrossEntropyLoss()(calibrated_logits, target_labels)
            
            # Check for NaN/inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected in loss at step {step}: {loss.item()}")
                break
                
            loss.backward()
            
            # Calculate gradient norm before clipping (like LogitsSharp)
            grad_norm = sum(p.grad.norm(p=2).item() for p in self.parameters() if p.grad is not None)
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()

            acc = (calibrated_logits.argmax(dim=1) == target_labels).float().mean()
            stats["loss"].append(loss.item())
            stats["acc"].append(acc.item())

            if verbose:
                pbar.set_description(f"loss {loss.item():.4e}, acc {acc.item():.4f}, gnorm {grad_norm:.4e}")
            
            # Debug: Print initial and final accuracy
            if step == 0:
                logger.debug(f"Initial accuracy: {acc.item():.4f}")
            elif step == max_steps - 1:
                logger.debug(f"Final accuracy: {acc.item():.4f}")

        # Calculate final calibrated probabilities and KL divergence
        with torch.no_grad():
            final_calibrated_probs = self.forward(ablated_probs)
            
            # Calculate mean of ablated probabilities across samples
            mean_ablated_probs = ablated_probs.mean(dim=0)
            
            # Calculate argmax ablated probabilities (one-hot) and their mean
            argmax_ablated_probs = torch.zeros_like(ablated_probs)
            argmax_ablated_indices = ablated_probs.argmax(dim=1)
            argmax_ablated_probs[torch.arange(ablated_probs.shape[0]), argmax_ablated_indices] = 1.0
            mean_argmaxed_ablated_probs = argmax_ablated_probs.mean(dim=0)
            
            # Calculate mean of final probabilities across samples
            mean_calibrated_probs = final_calibrated_probs.mean(dim=0)
            
            # Calculate argmax probabilities (one-hot) and their mean
            argmax_calibrated_probs = torch.zeros_like(final_calibrated_probs)
            argmax_indices = final_calibrated_probs.argmax(dim=1)
            argmax_calibrated_probs[torch.arange(final_calibrated_probs.shape[0]), argmax_indices] = 1.0
            mean_argmax_probs = argmax_calibrated_probs.mean(dim=0)
            
            # Create uniform distribution on the same device
            num_classes = final_calibrated_probs.shape[1]
            uniform_dist = torch.ones(num_classes, device=mean_calibrated_probs.device) / num_classes
            
            # Calculate KL divergences
            kl_div_probs = kl_divergence(mean_calibrated_probs, uniform_dist)
            kl_div_argmax = kl_divergence(mean_argmax_probs, uniform_dist)
            
            # Calculate accuracy with respect to target labels
            final_predictions = final_calibrated_probs.argmax(dim=1)
            accuracy = (final_predictions == target_labels).float().mean().item()
            
            # Debug information
            logger.debug(f"Target labels distribution: {torch.bincount(target_labels)}")
            logger.debug(f"Predictions distribution: {torch.bincount(final_predictions)}")
            logger.debug(f"Original predictions (before calibration): {torch.bincount(ablated_probs.argmax(dim=1))}")
            logger.debug(f"Sample target labels (first 10): {target_labels[:10]}")
            logger.debug(f"Sample predictions (first 10): {final_predictions[:10]}")
            logger.debug(f"Sample original preds (first 10): {ablated_probs.argmax(dim=1)[:10]}")
            
            # Extract learned weights and biases
            learned_params = {}
            if self.head_type == "scaling":
                # Scaling method: extract scale and bias parameters
                learned_params['scale'] = self.scale.cpu().numpy().tolist()
                learned_params['bias'] = self.bias.cpu().numpy().tolist()
            elif hasattr(self.head, 'weight') and hasattr(self.head, 'bias'):
                # Linear layer
                learned_params['weight'] = self.head.weight.cpu().numpy().tolist()
                learned_params['bias'] = self.head.bias.cpu().numpy().tolist() if self.head.bias is not None else None
            elif hasattr(self.head, 'module'):
                # ResidualBlock with sequential layers
                for name, layer in self.head.module.named_children():
                    if hasattr(layer, 'weight'):
                        learned_params[f'{name}_weight'] = layer.weight.cpu().numpy().tolist()
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            learned_params[f'{name}_bias'] = layer.bias.cpu().numpy().tolist()
            
            # Create comprehensive results dictionary
            results_data = {
                'probabilities': {
                    'mean_ablated_probabilities': mean_ablated_probs.cpu().numpy().tolist(),
                    'mean_argmaxed_ablated_probabilities': mean_argmaxed_ablated_probs.cpu().numpy().tolist(),
                    'mean_calibrated_probabilities': mean_calibrated_probs.cpu().numpy().tolist(),
                    'mean_argmax_probabilities': mean_argmax_probs.cpu().numpy().tolist(),
                    'uniform_distribution': uniform_dist.cpu().numpy().tolist()
                },
                'kl_divergences': {
                    'mean_probs_vs_uniform': float(kl_div_probs.item()),
                    'mean_argmax_vs_uniform': float(kl_div_argmax.item())
                },
                'accuracy': float(accuracy),
                'learned_parameters': learned_params
            }
            
            if fraction is not None:
                results_data = {
                    'fraction': fraction,
                    'experiment_id': experiment_id if experiment_id is not None else "default",
                    **results_data
                }
            self.fit_summary_ = results_data
            
            logger.debug(f"Final calibrated probabilities shape: {final_calibrated_probs.shape}")
            logger.debug(f"Mean ablated probabilities: {mean_ablated_probs}")
            logger.debug(f"Mean argmaxed ablated probabilities: {mean_argmaxed_ablated_probs}")
            logger.debug(f"Mean calibrated probabilities: {mean_calibrated_probs}")
            logger.debug(f"Mean argmax probabilities: {mean_argmax_probs}")
            logger.debug(f"Uniform distribution: {uniform_dist}")
            logger.debug(f"KL divergence (mean probs vs uniform): {kl_div_probs:.6f}")
            logger.debug(f"KL divergence (mean argmax vs uniform): {kl_div_argmax:.6f}")
            logger.debug(f"Accuracy (vs target labels): {accuracy:.6f}")


        self._is_fitted = True
        return stats
