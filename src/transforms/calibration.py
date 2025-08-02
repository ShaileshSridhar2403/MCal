"""Calibration transformation methods."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from .base import BaseTransform


class CalibrationTransform(BaseTransform):
    """Transform outputs using standard probability calibration methods.
    
    Supports:
    - Temperature scaling: divides logits by a learned temperature parameter
    - Platt scaling: applies a linear transformation to the logits
    """
    
    def __init__(
        self, 
        device: Optional[torch.device] = None, 
        method: str = 'temperature', 
        name: Optional[str] = None
    ):
        """Initialize calibration transform.
        
        Args:
            device: Device to run computations on
            method: Calibration method ('temperature' or 'platt')
            name: Optional name for the transform
        """
        super().__init__(device, name if name is not None else f"calibration_{method}")
        
        if method not in ['temperature', 'platt']:
            raise ValueError("method must be either 'temperature' or 'platt'")
            
        self.method = method
        self.temperature = None
        self.platt_weights = None
        self.platt_bias = None
    
    def fit(
        self, 
        tensor_path: str, 
        batch_size: int = 256, 
        num_epochs: int = 1000, 
        lr: float = 1e-3, 
        plot_calibration: bool = False, 
        save_path: str = "calibration_curve.png", 
        **kwargs
    ) -> Dict[str, Any]:
        """Fit calibration parameters to the data.
        
        Args:
            tensor_path: Path to input tensor file
            batch_size: Batch size for optimization
            num_epochs: Number of training epochs
            lr: Learning rate
            plot_calibration: Whether to plot calibration curves
            save_path: Path to save calibration curve plot
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with training statistics
        """
        full_tensor = np.load(tensor_path)
        n_fractions, n_samples, n_outputs = full_tensor.shape
        
        # Initialize parameters for each fraction
        if self.method == 'temperature':
            self.temperature = torch.ones(n_fractions, device=self.device)
        elif self.method == 'platt':
            self.platt_weights = torch.ones((n_fractions, n_outputs), device=self.device)
            self.platt_bias = torch.zeros((n_fractions, n_outputs), device=self.device)
        
        # Split data for training and validation
        split_idx = int(0.8 * n_samples)
        stats = {"final_temperatures": [], "final_losses": []}
        
        for fraction in tqdm(range(n_fractions), desc="Calibrating probabilities"):
            predictions = full_tensor[fraction]
            
            # Split data
            train_preds = predictions[:split_idx]
            val_preds = predictions[split_idx:]
            
            # Generate pseudo-labels (one-hot encoding of predicted class)
            train_pred_classes = np.argmax(train_preds, axis=1)
            val_pred_classes = np.argmax(val_preds, axis=1)
            
            if self.method == 'temperature':
                final_temp, final_loss = self._fit_temperature(
                    fraction, train_preds, train_pred_classes, 
                    val_preds, val_pred_classes, num_epochs, lr
                )
                stats["final_temperatures"].append(final_temp)
                stats["final_losses"].append(final_loss)
            elif self.method == 'platt':
                final_loss = self._fit_platt(
                    fraction, train_preds, train_pred_classes,
                    val_preds, val_pred_classes, batch_size, num_epochs, lr
                )
                stats["final_losses"].append(final_loss)
        
        if plot_calibration:
            self._plot_calibration_curves(full_tensor, save_path)
        
        self._is_fitted = True
        return stats
    
    def _fit_temperature(
        self, 
        fraction: int, 
        train_preds: np.ndarray, 
        train_classes: np.ndarray,
        val_preds: np.ndarray, 
        val_classes: np.ndarray, 
        num_epochs: int, 
        lr: float
    ) -> tuple:
        """Fit temperature scaling parameters."""
        train_preds_tensor = torch.tensor(train_preds, dtype=torch.float32, device=self.device)
        val_preds_tensor = torch.tensor(val_preds, dtype=torch.float32, device=self.device)
        train_classes_tensor = torch.tensor(train_classes, dtype=torch.long, device=self.device)
        val_classes_tensor = torch.tensor(val_classes, dtype=torch.long, device=self.device)
        
        # Convert probabilities to logits
        train_logits = torch.log(train_preds_tensor.clamp(1e-8, 1-1e-8))
        val_logits = torch.log(val_preds_tensor.clamp(1e-8, 1-1e-8))
        
        # Initialize temperature parameter
        temperature = torch.tensor([1.0], requires_grad=True, device=self.device)
        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=num_epochs)
        
        nll_criterion = nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_temp = 1.0
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = val_logits / temperature
            loss = nll_criterion(scaled_logits, val_classes_tensor)
            loss.backward()
            return loss
        
        # Training loop
        for _ in range(min(num_epochs, 50)):  # LBFGS typically converges fast
            optimizer.step(eval_loss)
            
            with torch.no_grad():
                scaled_logits = val_logits / temperature
                val_loss = nll_criterion(scaled_logits, val_classes_tensor)
                
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_temp = temperature.item()
        
        self.temperature[fraction] = best_temp
        return best_temp, best_loss
    
    def _fit_platt(
        self, 
        fraction: int, 
        train_preds: np.ndarray, 
        train_classes: np.ndarray,
        val_preds: np.ndarray, 
        val_classes: np.ndarray, 
        batch_size: int, 
        num_epochs: int, 
        lr: float
    ) -> float:
        """Fit Platt scaling parameters."""
        train_preds_tensor = torch.tensor(train_preds, dtype=torch.float32, device=self.device)
        val_preds_tensor = torch.tensor(val_preds, dtype=torch.float32, device=self.device)
        train_classes_tensor = torch.tensor(train_classes, dtype=torch.long, device=self.device)
        val_classes_tensor = torch.tensor(val_classes, dtype=torch.long, device=self.device)
        
        # Convert probabilities to logits
        train_logits = torch.log(train_preds_tensor.clamp(1e-8, 1-1e-8))
        val_logits = torch.log(val_preds_tensor.clamp(1e-8, 1-1e-8))
        
        n_outputs = train_preds.shape[1]
        
        # Initialize parameters
        weights = torch.ones(n_outputs, requires_grad=True, device=self.device)
        bias = torch.zeros(n_outputs, requires_grad=True, device=self.device)
        
        optimizer = optim.Adam([weights, bias], lr=lr)
        nll_criterion = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        best_weights = weights.clone().detach()
        best_bias = bias.clone().detach()
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            
            # Process in batches
            for i in range(0, len(train_preds), batch_size):
                batch_logits = train_logits[i:i+batch_size]
                batch_classes = train_classes_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Apply Platt scaling
                scaled_logits = batch_logits * weights + bias
                loss = nll_criterion(scaled_logits, batch_classes)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Evaluate on validation set
            with torch.no_grad():
                scaled_val_logits = val_logits * weights + bias
                val_loss = nll_criterion(scaled_val_logits, val_classes_tensor)
                
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_weights = weights.clone().detach()
                    best_bias = bias.clone().detach()
        
        # Store best parameters
        self.platt_weights[fraction] = best_weights
        self.platt_bias[fraction] = best_bias
        
        return best_loss
    
    def _plot_calibration_curves(self, full_tensor: np.ndarray, save_path: str) -> None:
        """Plot reliability diagrams (calibration curves)."""
        n_fractions = full_tensor.shape[0]
        n_cols = min(4, n_fractions)
        n_rows = (n_fractions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for fraction in range(min(n_fractions, len(axes))):
            ax = axes[fraction]
            
            # Get original and calibrated predictions
            predictions = full_tensor[fraction]
            calibrated = self.transform(full_tensor[fraction:fraction+1])[0]
            
            # Simple binning for reliability diagram
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            # Calculate bin statistics for original predictions
            max_probs_orig = np.max(predictions, axis=1)
            bin_indices = np.digitize(max_probs_orig, bin_boundaries) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Calculate bin statistics for calibrated predictions
            max_probs_cal = np.max(calibrated, axis=1)
            bin_indices_cal = np.digitize(max_probs_cal, bin_boundaries) - 1
            bin_indices_cal = np.clip(bin_indices_cal, 0, n_bins - 1)
            
            # Plot calibration curve
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
            
            # Simple visualization showing confidence distribution
            ax.hist(max_probs_orig, bins=n_bins, alpha=0.3, label='Original', density=True)
            ax.hist(max_probs_cal, bins=n_bins, alpha=0.3, label='Calibrated', density=True)
            
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Density')
            ax.set_title(f'Confidence Distribution - Fraction {fraction}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for j in range(n_fractions, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply calibration transformation to input tensor."""
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")
        
        if isinstance(input_tensor, np.ndarray):
            transformed_outputs = np.zeros_like(input_tensor)
            
            for fraction in range(input_tensor.shape[0]):
                predictions = torch.tensor(
                    input_tensor[fraction], 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                if self.method == 'temperature':
                    # Temperature scaling
                    logits = torch.log(predictions.clamp(1e-8, 1-1e-8))
                    temperature = self.temperature[fraction]
                    calibrated_outputs = torch.softmax(logits / temperature, dim=1)
                elif self.method == 'platt':
                    # Platt scaling
                    logits = torch.log(predictions.clamp(1e-8, 1-1e-8))
                    weights = self.platt_weights[fraction]
                    bias = self.platt_bias[fraction]
                    calibrated_outputs = torch.softmax(logits * weights + bias, dim=1)
                
                transformed_outputs[fraction] = calibrated_outputs.cpu().numpy()
            
            return transformed_outputs
        else:
            # Handle single tensor case
            predictions = input_tensor.to(self.device)
            
            if self.method == 'temperature':
                logits = torch.log(predictions.clamp(1e-8, 1-1e-8))
                temperature = self.temperature[0] if len(self.temperature) > 0 else 1.0
                return torch.softmax(logits / temperature, dim=1)
            elif self.method == 'platt':
                logits = torch.log(predictions.clamp(1e-8, 1-1e-8))
                weights = self.platt_weights[0] if len(self.platt_weights) > 0 else torch.ones_like(logits[0])
                bias = self.platt_bias[0] if len(self.platt_bias) > 0 else torch.zeros_like(logits[0])
                return torch.softmax(logits * weights + bias, dim=1)
    
    def save(self, path: str) -> None:
        """Save calibration parameters."""
        save_dict = {'method': self.method}
        
        if self.method == 'temperature' and self.temperature is not None:
            save_dict['temperature'] = self.temperature.cpu().numpy()
        elif self.method == 'platt' and self.platt_weights is not None:
            save_dict['platt_weights'] = self.platt_weights.cpu().numpy()
            save_dict['platt_bias'] = self.platt_bias.cpu().numpy()
        
        np.save(path, save_dict)
    
    def load(self, path: str) -> None:
        """Load calibration parameters."""
        save_dict = np.load(path, allow_pickle=True).item()
        
        self.method = save_dict['method']
        
        if self.method == 'temperature' and 'temperature' in save_dict:
            self.temperature = torch.tensor(save_dict['temperature'], device=self.device)
        elif self.method == 'platt' and 'platt_weights' in save_dict:
            self.platt_weights = torch.tensor(save_dict['platt_weights'], device=self.device)
            self.platt_bias = torch.tensor(save_dict['platt_bias'], device=self.device)
        
        self._is_fitted = True