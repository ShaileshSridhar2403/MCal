"""Optimization utilities for probability calibration."""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def get_expectation(
    function_inputs: torch.Tensor, 
    device: Optional[torch.device] = None, 
    normalization: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate expectation of probability distributions.
    
    Args:
        function_inputs: Input probability distributions of shape (batch_size, num_classes)
        device: Device to perform computations on
        normalization: Whether to normalize the expectations
        
    Returns:
        Tuple of (one_hot_expectation, prob_expectation)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    function_inputs = function_inputs.to(device)

    # Calculate one-hot expectation
    one_hot_outputs = make_one_hot(function_inputs)
    one_hot_expectation = one_hot_outputs.mean(dim=0)

    # Handle zero values in one_hot_expectation
    epsilon = 1e-9
    zero_indices = torch.where(one_hot_expectation == 0)[0]
    if len(zero_indices) > 0:
        max_index = torch.argmax(one_hot_expectation)
        for zero_index in zero_indices:
            one_hot_expectation[zero_index] = epsilon
            one_hot_expectation[max_index] -= epsilon

    # Normalize if requested
    if normalization:
        one_hot_expectation /= one_hot_expectation.sum()

    # Calculate probability expectation
    prob_expectation = function_inputs.mean(dim=0)
    
    if normalization:
        prob_expectation = prob_expectation / prob_expectation.sum()
    
    return one_hot_expectation, prob_expectation


def make_one_hot(outputs: torch.Tensor) -> torch.Tensor:
    """Convert probability distributions to one-hot encoding based on argmax.
    
    Args:
        outputs: Probability distributions of shape (batch_size, num_classes)
        
    Returns:
        One-hot encoded tensor of same shape
    """
    max_indices = torch.argmax(outputs, dim=1)
    num_columns = outputs.shape[1]
    one_hot_matrix = torch.eye(num_columns, device=outputs.device)[max_indices]
    return one_hot_matrix


def kl_divergence(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence between two probability distributions.
    
    Args:
        P: First probability distribution
        Q: Second probability distribution
        
    Returns:
        KL divergence D(P||Q)
    """
    eps = 1e-8
    P = P.clamp(min=eps, max=1-eps)
    Q = Q.clamp(min=eps, max=1-eps)
    return (P * (P / Q).log()).sum()


def apply_lambda_adjustment(
    input_tensor: torch.Tensor, 
    lambdas: torch.Tensor, 
    device: Optional[torch.device] = None, 
    normalization: bool = False
) -> torch.Tensor:
    """Apply lambda scaling adjustment to probability distributions.
    
    Args:
        input_tensor: Input probability distributions
        lambdas: Lambda scaling parameters
        device: Device to perform computations on
        normalization: Whether to normalize after scaling
        
    Returns:
        Lambda-adjusted probability distributions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors if needed
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    if isinstance(lambdas, np.ndarray):
        lambdas = torch.tensor(lambdas, dtype=torch.float32)

    # Move to device
    input_tensor = input_tensor.to(device)
    lambdas = lambdas.to(device)

    # Apply lambda adjustment
    if lambdas.dim() == 1:
        # If lambdas is 1D, we're applying the same lambdas to all samples
        scaled_outputs = input_tensor * lambdas.unsqueeze(0)
    else:
        # If lambdas is 2D, we have different lambdas for each fraction/sample
        scaled_outputs = input_tensor * lambdas

    # Normalize if requested
    if normalization:
        scaled_outputs = scaled_outputs / scaled_outputs.sum(dim=-1, keepdim=True)

    return scaled_outputs


def find_optimal_lambda_batch(
    predictions_augmented: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    num_epochs: int = 5000,
    save_path: str = "loss_curve.png",
    initial_lambda_vars: Optional[torch.Tensor] = None,
    n_starts: int = 1,
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, float]:
    """Find optimal lambda values using batch optimization.
    
    Args:
        predictions_augmented: Input predictions to optimize
        device: Device to run optimization on
        batch_size: Batch size for optimization
        num_epochs: Number of optimization epochs
        save_path: Path to save loss curve plot
        initial_lambda_vars: Initial lambda values (optional)
        n_starts: Number of random starts
        
    Returns:
        Tuple of (best_lambdas, loss1, one_hot_expectation, prob_expectation, final_loss)
    """
    n_samples, n_outputs = predictions_augmented.shape
    
    best_result = None
    best_loss = float('inf')
    
    for start in range(n_starts):
        if initial_lambda_vars is not None and start == 0:
            lambda_vars = initial_lambda_vars.clone().detach().to(device)
        else:
            lambda_vars = torch.ones(n_outputs, device=device)
        
        lambda_vars.requires_grad = True
        predictions_augmented = predictions_augmented.to(device)
        
        U = torch.full((n_outputs,), 1/n_outputs, device=device)
        optimizer = optim.Adam([lambda_vars], lr=0.01)
        loss_history = []
        
        for epoch in tqdm(range(num_epochs), desc=f"Start {start+1}/{n_starts}"):
            running_loss = 0

            shuffle_indices = torch.randperm(n_samples)
            predictions_shuffled = predictions_augmented[shuffle_indices]

            for i in range(0, n_samples, batch_size):
                batch_predictions = predictions_shuffled[i:i+batch_size]
                
                scaled_m = batch_predictions * lambda_vars
                scaled_m = scaled_m / scaled_m.sum(dim=1, keepdim=True)
                
                E_m = scaled_m.mean(dim=0)
                corrected_E_m = E_m / E_m.sum()
                
                loss1 = kl_divergence(corrected_E_m, U)
                batch_loss = loss1

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                running_loss += batch_loss.item()

                with torch.no_grad():
                    lambda_vars.data = torch.clamp(lambda_vars.data, min=0)
            
            avg_loss = running_loss / (n_samples // batch_size + 1)
            loss_history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_result = (
                    lambda_vars.clone().detach(),
                    loss1.item(),
                    make_one_hot(scaled_m).mean(dim=0).detach(),
                    corrected_E_m.detach(),
                    batch_loss.item()
                )
        
        # Save loss curve for first start
        if start == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss vs. Epoch')
            plt.savefig(save_path)
            plt.close()
    
    return best_result