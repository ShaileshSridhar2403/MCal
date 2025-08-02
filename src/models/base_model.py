"""Base model classes for different modalities."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union, List
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC, nn.Module):
    """Abstract base class for all models.
    
    Provides common functionality for model training, inference, and evaluation
    across different modalities (vision, language, tabular).
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "base_model",
        device: Optional[torch.device] = None
    ):
        """Initialize base model.
        
        Args:
            num_classes: Number of output classes
            model_name: Name of the model
            device: Device to run model on
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.to(self.device)
        
        # Model metadata
        self.model_info = {
            "name": model_name,
            "num_classes": num_classes,
            "device": str(self.device)
        }
        
        # Training state
        self.is_trained = False
        self.training_history = {}
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (logits)
        """
        pass
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor (after softmax)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = self.model_info.copy()
        info.update({
            "is_trained": self.is_trained,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        })
        return info
    
    def save_model(self, path: str) -> None:
        """Save model state dict.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'training_history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model state dict.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.model_info.update(checkpoint.get('model_info', {}))
        self.training_history = checkpoint.get('training_history', {})
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}', num_classes={self.num_classes})"


class ExpectationTrackingModel(BaseModel):
    """Model wrapper that tracks expectation values for calibration.
    
    This is similar to the NormalModel in XAI_Benchmark but more general.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        model_name: str = "expectation_tracking_model",
        device: Optional[torch.device] = None
    ):
        """Initialize expectation tracking model.
        
        Args:
            base_model: Underlying model to wrap
            num_classes: Number of output classes
            model_name: Name of the model
            device: Device to run model on
        """
        super().__init__(num_classes, model_name, device)
        self.base_model = base_model.to(self.device)
        self.expectation_store = []
        self.reset_expectation_store()
    
    def reset_expectation_store(self) -> None:
        """Reset the expectation store."""
        self.expectation_store = []
    
    def get_expectation_store_expectation(self) -> np.ndarray:
        """Get probability expectation from stored values.
        
        Returns:
            Mean expectation across all stored batches
        """
        if not self.expectation_store:
            logger.warning("Expectation store is empty")
            return np.zeros(self.num_classes)
        
        return np.vstack(self.expectation_store).mean(axis=0)
    
    def get_expectation_store_onehot_expectation(self) -> np.ndarray:
        """Get one-hot expectation from stored values.
        
        Returns:
            Mean one-hot expectation across all stored batches
        """
        if not self.expectation_store:
            logger.warning("Expectation store is empty")
            return np.zeros(self.num_classes)
        
        # Convert stored expectations to tensor
        preds_tensor = torch.tensor(np.vstack(self.expectation_store))
        
        # Get one-hot vectors
        one_hot_preds = self.make_one_hot(preds_tensor)
        
        # Return mean of one-hot vectors
        return one_hot_preds.mean(dim=0).cpu().numpy()
    
    def make_one_hot(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert probability outputs to one-hot encoding.
        
        Args:
            outputs: Probability outputs of shape (batch_size, num_classes)
            
        Returns:
            One-hot encoded tensor
        """
        max_indices = torch.argmax(outputs, dim=1)
        one_hot_matrix = torch.eye(self.num_classes, device=outputs.device)[max_indices]
        return one_hot_matrix
    
    def forward(self, x: torch.Tensor, track_expectations: bool = True) -> torch.Tensor:
        """Forward pass with expectation tracking.
        
        Args:
            x: Input tensor
            track_expectations: Whether to track expectations in store
            
        Returns:
            Softmax probabilities
        """
        # Get outputs from base model
        outputs = self.base_model(x)
        
        # Apply softmax
        softmaxed_outputs = torch.softmax(outputs, dim=1)
        
        # Store batch-level expectation if tracking
        if track_expectations and self.training:
            batch_expectation = torch.mean(softmaxed_outputs, dim=0).detach().cpu().numpy()
            self.expectation_store.append(batch_expectation)
        
        return softmaxed_outputs
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions without tracking expectations."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, track_expectations=False)


class ModelWrapper(BaseModel):
    """Generic wrapper for pre-trained models."""
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        model_name: str = "wrapped_model",
        device: Optional[torch.device] = None,
        preprocess_fn: Optional[callable] = None
    ):
        """Initialize model wrapper.
        
        Args:
            base_model: Pre-trained model to wrap
            num_classes: Number of output classes
            model_name: Name of the model
            device: Device to run model on
            preprocess_fn: Optional preprocessing function
        """
        super().__init__(num_classes, model_name, device)
        self.base_model = base_model.to(self.device)
        self.preprocess_fn = preprocess_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through wrapped model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model outputs (logits)
        """
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        
        return self.base_model(x)


class EnsembleModel(BaseModel):
    """Ensemble of multiple models."""
    
    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int,
        model_name: str = "ensemble_model",
        device: Optional[torch.device] = None,
        voting: str = "soft"
    ):
        """Initialize ensemble model.
        
        Args:
            models: List of models to ensemble
            num_classes: Number of output classes
            model_name: Name of the ensemble
            device: Device to run models on
            voting: Voting method ('soft' for probability averaging, 'hard' for majority vote)
        """
        super().__init__(num_classes, model_name, device)
        self.models = nn.ModuleList([model.to(self.device) for model in models])
        self.voting = voting
        self.num_models = len(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble outputs
        """
        if self.voting == "soft":
            # Average probabilities
            probs_list = []
            for model in self.models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
            
            avg_probs = torch.stack(probs_list).mean(dim=0)
            # Convert back to logits for consistency
            return torch.log(avg_probs + 1e-8)
        
        else:  # hard voting
            # Majority vote on predictions
            votes = []
            for model in self.models:
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                votes.append(preds)
            
            votes_tensor = torch.stack(votes, dim=1)  # (batch_size, num_models)
            
            # Get majority vote for each sample
            batch_size = x.shape[0]
            ensemble_preds = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            for i in range(batch_size):
                unique_votes, counts = torch.unique(votes_tensor[i], return_counts=True)
                majority_vote = unique_votes[torch.argmax(counts)]
                ensemble_preds[i] = majority_vote
            
            # Convert to one-hot logits
            ensemble_logits = torch.zeros(batch_size, self.num_classes, device=self.device)
            ensemble_logits[range(batch_size), ensemble_preds] = 1.0
            
            return ensemble_logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble probability predictions."""
        self.eval()
        with torch.no_grad():
            if self.voting == "soft":
                # Average probabilities directly
                probs_list = []
                for model in self.models:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(x)
                    else:
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)
                    probs_list.append(probs)
                
                return torch.stack(probs_list).mean(dim=0)
            else:
                # Use forward pass and convert to probabilities
                logits = self.forward(x)
                return torch.softmax(logits, dim=1)