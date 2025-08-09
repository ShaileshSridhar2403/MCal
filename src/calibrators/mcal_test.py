"""MCal - Vector scaling calibration model."""

from typing import Optional, Dict, Any
from experiments.ilovekldiv import calibrated_logits_hybrid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseCalibrator

def kl_divergence_batch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Computes the KL divergence for a batch of probability distributions.

    Calculates D_KL(P || Q) for each row in the batch, then returns the average.

    Args:
        p (torch.Tensor): Batch of probability distributions, shape (N, d).
        q (torch.Tensor): Batch of probability distributions, shape (N, d).
        epsilon (float): A small value to ensure numerical stability.

    Returns:
        torch.Tensor: The average KL divergence over the batch (scalar).
    """
    # Clamp for numerical stability to avoid log(0)
    p_stable = torch.clamp(p, min=eps)
    q_stable = torch.clamp(q, min=eps)

    # Element-wise operation, result shape (N, d)
    # This is p * (log(p) - log(q)) for each element
    element_wise_div = p_stable * (p_stable.log() - q_stable.log())

    # Sum along dimension 1 to get KL divergence for each item in the batch
    # This results in a tensor of shape (N,)
    sum_over_dist = element_wise_div.sum(dim=-1)

    # Average the KL divergences across the batch to get a single scalar value
    average_kl = sum_over_dist.mean()

    return average_kl


class MCal_Test(BaseCalibrator):
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
        ablated_probs: torch.Tensor | None = None,
        target_distribution: torch.Tensor | None = None,
        loss_mode: str = "KL_Exp"
    ):
        super().__init__(num_classes)
        self.w = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.loss_mode = loss_mode

        if ablated_probs is not None:
            self.fit(ablated_probs)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the calibration model.
        
        Args:
            probs (torch.Tensor): Input probability distributions of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: Calibrated probability distributions
        """
        self._validate_input_probs(probs)
        
        probs_clamped = torch.clamp(probs, min=1e-8, max=1-1e-8)
        logits = torch.log(probs_clamped)
        logits = self.w.view(1, -1) * logits + self.b.view(1, -1)
        q = F.softmax(logits, dim=1)
        return q

    def fit(
        self,
        ablated_probs: torch.Tensor,
        target_probs: torch.Tensor | None = None,
        kappa: float = 1.0,
        max_steps: int = 5000,
        lr: float = 1e-2,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Fit the calibration model to the given probability distributions.
        
        Args:
            ablated_probs (torch.Tensor): Ablated probability distributions
            target_probs (Optional[torch.Tensor]): Target probabilities to optimize towards (defaults to uniform)
            kappa (float): Sharpening factor for probability distributions
            max_steps (int): Maximum number of optimization steps
            lr (float): Learning rate for optimization
            verbose (bool): Whether to show progress bar and metrics
            
        Returns:
            Dictionary containing training statistics
        """

        if target_probs is None:
            target_probs = torch.ones_like(ablated_probs) / self.num_classes

        ablated_probs = ablated_probs.clamp(min=1e-8, max=1-1e-8)
        target_probs = target_probs.clamp(min=1e-8, max=1-1e-8)
        target_classes = target_probs.argmax(dim=1, keepdim=True).float()

        stats = {"loss": [], "acc": []}

        optimizer = optim.Adam(self.parameters(), lr=lr)

        pbar = tqdm(range(max_steps), desc="MCal Training") if verbose else range(max_steps)
        for _ in pbar:
            optimizer.zero_grad()

            q = self.forward(ablated_probs)
            s = (q / q.max(dim=1, keepdim=True).values)
            s = s ** kappa
            s = s / s.sum(dim=1, keepdim=True)
            
            if self.loss_mode == "KL_Exp":
                loss = kl_divergence_batch(s.mean(dim=0, keepdim=True), target_classes.mean(dim=0, keepdim=True))
            elif self.loss_mode == "Exp_KL":
                loss = kl_divergence_batch(s, target_classes)
            elif self.loss_mode == "CE":
                # For CE loss, we need integer class labels
                target_labels = target_probs.argmax(dim=1)
                loss = nn.CrossEntropyLoss()(s, target_labels)
            else:
                raise ValueError(f"Invalid loss mode: {self.loss_mode}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate accuracy based on argmax of target probs
            target_labels = target_probs.argmax(dim=1)
            acc = (s.argmax(dim=1) == target_labels).float().mean()

            stats["loss"].append(loss.item())
            stats["acc"].append(acc.item())

            if verbose:
                pbar.set_description(f"Mode {self.loss_mode},  Loss {loss.item():.3e}, Acc {acc:.3f}")

        self._is_fitted = True
        return stats


class MCal_CE(BaseCalibrator):
    """MCal with cross-entropy loss."""

    def __init__(self, num_classes: int, head_type: str = "linear"):
        super().__init__(num_classes)

        self.head_type = head_type
        if head_type == "linear":
            self.head = nn.Linear(num_classes, num_classes)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(num_classes, 8 * num_classes),
                nn.GELU(),
                nn.Linear(8 * num_classes, num_classes))
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def forward(self, ablated_probs: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass of the calibration model.
        
        Args:
            ablated_probs (torch.Tensor): Ablated probability distributions
            return_logits (bool): Whether to return logits or probabilities
            
        Returns:
            torch.Tensor: Calibrated probability distributions
        """
        ablated_logits = torch.log(ablated_probs.clamp(min=1e-8))
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
    ) -> Dict[str, Any]:
        """Fit the calibration model to the given probability distributions.
        
        Args:
            ablated_probs (torch.Tensor): Ablated probability distributions
            target_labels (torch.Tensor): Target labels
            max_steps (int): Maximum number of optimization steps
            lr (float): Learning rate for optimization
            verbose (bool): Whether to show progress bar and metrics
            
        Returns:
            Dictionary containing training statistics
        """

        optimizer = optim.Adam(self.parameters(), lr=lr)
        stats = {"loss": [], "acc": []}

        pbar = tqdm(range(max_steps), desc="MCal Training") if verbose else range(max_steps)
        for _ in pbar:
            optimizer.zero_grad()

            calibrated_logits = self.forward(ablated_probs, return_logits=True)
            loss = nn.CrossEntropyLoss()(calibrated_logits, target_labels)
            loss.backward()
            optimizer.step()

            acc = (calibrated_logits.argmax(dim=1) == target_labels).float().mean()
            stats["loss"].append(loss.item())
            stats["acc"].append(acc.item())

            if verbose:
                pbar.set_description(f"Head {self.head_type}, Loss {loss.item():.3e}, Acc {acc:.3f}")

        self._is_fitted = True
        return stats

