"""Tabular models for structured data classification."""

from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
import logging

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class MLPClassifier(BaseModel):
    """Multi-layer perceptron for tabular classification."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
        model_name: str = "mlp_classifier",
        device: Optional[torch.device] = None
    ):
        """Initialize MLP classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            model_name: Name of the model
            device: Device to run model on
        """
        super().__init__(num_classes, model_name, device)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        self.layers = self._build_layers()
    
    def _build_layers(self) -> nn.ModuleList:
        """Build MLP layers."""
        layers = nn.ModuleList()
        
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        for layer in self.layers:
            x = layer(x)
        
        return x


class XGBoostWrapper(BaseModel):
    """Wrapper for XGBoost classifier."""
    
    def __init__(
        self,
        num_classes: int,
        input_dim: Optional[int] = None,
        model_name: str = "xgboost_classifier",
        device: Optional[torch.device] = None,
        **xgb_params
    ):
        """Initialize XGBoost wrapper.
        
        Args:
            num_classes: Number of output classes
            input_dim: Number of input features (optional)
            model_name: Name of the model
            device: Device (XGBoost runs on CPU)
            **xgb_params: XGBoost parameters
        """
        # XGBoost runs on CPU
        device = torch.device("cpu")
        super().__init__(num_classes, model_name, device)
        
        self.input_dim = input_dim
        self.xgb_params = xgb_params
        self.model = None
        
        # Set default parameters
        default_params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        # Update with user parameters
        for key, value in default_params.items():
            if key not in self.xgb_params:
                self.xgb_params[key] = value
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize XGBoost model."""
        try:
            import xgboost as xgb
            
            if self.num_classes > 2:
                self.model = xgb.XGBClassifier(**self.xgb_params)
            else:
                # Remove num_class for binary classification
                params = self.xgb_params.copy()
                params.pop('num_class', None)
                self.model = xgb.XGBClassifier(**params)
            
            logger.info("Initialized XGBoost classifier")
            
        except ImportError:
            logger.error("xgboost library not found. Install with: pip install xgboost")
            raise
    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[List[Tuple]] = None) -> None:
        """Fit XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Evaluation set for early stopping
        """
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        self.is_trained = True
        logger.info("XGBoost model training completed")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (for compatibility with PyTorch interface).
        
        Args:
            x: Input tensor
            
        Returns:
            Probability predictions as logits
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        
        # Get probabilities
        probs = self.model.predict_proba(x_np)
        
        # Convert back to tensor and return as logits
        probs_tensor = torch.from_numpy(probs).float()
        
        # Convert probabilities to logits for consistency
        logits = torch.log(probs_tensor + 1e-8)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        x_np = x.detach().cpu().numpy()
        probs = self.model.predict_proba(x_np)
        return torch.from_numpy(probs).float()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        x_np = x.detach().cpu().numpy()
        preds = self.model.predict(x_np)
        return torch.from_numpy(preds).long()


class TabTransformerWrapper(BaseModel):
    """Wrapper for TabTransformer model."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        ff_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        model_name: str = "tab_transformer",
        device: Optional[torch.device] = None
    ):
        """Initialize TabTransformer.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            ff_dropout: Feedforward dropout
            attn_dropout: Attention dropout
            model_name: Name of the model
            device: Device to run model on
        """
        super().__init__(num_classes, model_name, device)
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(ff_dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Project to embedding dimension
        x = self.input_projection(x)  # (batch_size, embed_dim)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Pass through transformer
        x = self.transformer(x)  # (batch_size, 1, embed_dim)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch_size, embed_dim)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class TabPFNWrapper(BaseModel):
    """Wrapper for TabPFN (Tabular Prior-Data Fitted Network)."""
    
    def __init__(
        self,
        num_classes: int,
        input_dim: Optional[int] = None,
        model_name: str = "tabpfn",
        device: Optional[torch.device] = None
    ):
        """Initialize TabPFN wrapper.
        
        Args:
            num_classes: Number of output classes
            input_dim: Number of input features
            model_name: Name of the model
            device: Device to run model on
        """
        super().__init__(num_classes, model_name, device)
        
        self.input_dim = input_dim
        self.model = None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize TabPFN model."""
        try:
            from tabpfn import TabPFNClassifier
            
            self.model = TabPFNClassifier(device=str(self.device))
            logger.info("Initialized TabPFN classifier")
            
        except ImportError:
            logger.error("tabpfn library not found. Install with: pip install tabpfn")
            self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit TabPFN model.
        
        Args:
            X: Training features
            y: Training labels
        """
        if self.model is None:
            logger.warning("TabPFN not available, skipping training")
            return
        
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("TabPFN model training completed")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.model is None:
            raise RuntimeError("TabPFN not available")
        
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        x_np = x.detach().cpu().numpy()
        probs = self.model.predict_proba(x_np)
        
        # Convert to logits
        probs_tensor = torch.from_numpy(probs).float()
        logits = torch.log(probs_tensor + 1e-8)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        if self.model is None:
            raise RuntimeError("TabPFN not available")
        
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        x_np = x.detach().cpu().numpy()
        probs = self.model.predict_proba(x_np)
        return torch.from_numpy(probs).float()


# Registry of tabular models
TABULAR_MODELS = {
    "mlp": MLPClassifier,
    "xgboost": XGBoostWrapper,
    "tab_transformer": TabTransformerWrapper,
    "tabpfn": TabPFNWrapper,
}


def get_tabular_model(model_name: str, **kwargs) -> BaseModel:
    """Get a tabular model by name.
    
    Args:
        model_name: Name of the tabular model
        **kwargs: Additional arguments
        
    Returns:
        Tabular model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in TABULAR_MODELS:
        raise ValueError(
            f"Unknown tabular model: {model_name}. "
            f"Available models: {list(TABULAR_MODELS.keys())}"
        )
    
    return TABULAR_MODELS[model_name](**kwargs)