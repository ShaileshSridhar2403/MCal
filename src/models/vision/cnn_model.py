"""Convolutional Neural Network models for vision tasks."""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class BasicCNN(BaseModel):
    """Basic CNN architecture for image classification."""
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        input_size: int = 224,
        model_name: str = "basic_cnn",
        device: Optional[torch.device] = None
    ):
        """Initialize basic CNN.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            input_size: Input image size (assumed square)
            model_name: Name of the model
            device: Device to run model on
        """
        super().__init__(num_classes, model_name, device)
        
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate feature size after convolutions
        self.feature_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _get_conv_output_size(self) -> int:
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional features
        x = self.conv_layers(x)
        
        # Flatten for classifier
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class ResNetWrapper(BaseModel):
    """Wrapper for torchvision ResNet models."""
    
    def __init__(
        self,
        num_classes: int,
        architecture: str = "resnet18",
        pretrained: bool = True,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize ResNet wrapper.
        
        Args:
            num_classes: Number of output classes
            architecture: ResNet architecture ('resnet18', 'resnet34', 'resnet50', etc.)
            pretrained: Whether to use pretrained weights
            model_name: Name of the model
            device: Device to run model on
        """
        if model_name is None:
            model_name = f"{architecture}_{'pretrained' if pretrained else 'scratch'}"
        
        super().__init__(num_classes, model_name, device)
        
        self.architecture = architecture
        self.pretrained = pretrained
        
        # Get ResNet model
        if hasattr(models, architecture):
            if pretrained:
                self.backbone = getattr(models, architecture)(weights='DEFAULT')
            else:
                self.backbone = getattr(models, architecture)(weights=None)
        else:
            raise ValueError(f"Unknown ResNet architecture: {architecture}")
        
        # Replace final layer
        if hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        
        # Freeze backbone if desired
        self.frozen = False
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze backbone parameters.
        
        Args:
            freeze: Whether to freeze backbone parameters
        """
        self.frozen = freeze
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always allow final layer to be trainable
        if hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.backbone, 'classifier'):
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        
        logger.info(f"Backbone {'frozen' if freeze else 'unfrozen'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.backbone(x)


class EfficientNetWrapper(BaseModel):
    """Wrapper for EfficientNet models."""
    
    def __init__(
        self,
        num_classes: int,
        architecture: str = "efficientnet_b0",
        pretrained: bool = True,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize EfficientNet wrapper.
        
        Args:
            num_classes: Number of output classes
            architecture: EfficientNet architecture
            pretrained: Whether to use pretrained weights
            model_name: Name of the model
            device: Device to run model on
        """
        if model_name is None:
            model_name = f"{architecture}_{'pretrained' if pretrained else 'scratch'}"
        
        super().__init__(num_classes, model_name, device)
        
        self.architecture = architecture
        self.pretrained = pretrained
        
        # Get EfficientNet model
        try:
            if pretrained:
                self.backbone = getattr(models, architecture)(weights='DEFAULT')
            else:
                self.backbone = getattr(models, architecture)(weights=None)
        except AttributeError:
            raise ValueError(f"Unknown EfficientNet architecture: {architecture}")
        
        # Replace classifier
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
        
        self.frozen = False
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze backbone parameters."""
        self.frozen = freeze
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = not freeze
        
        logger.info(f"EfficientNet backbone {'frozen' if freeze else 'unfrozen'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EfficientNet."""
        return self.backbone(x)


class CustomCNN(BaseModel):
    """Customizable CNN with configurable architecture."""
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        input_size: int = 224,
        conv_layers: List[Dict[str, Any]] = None,
        fc_layers: List[int] = None,
        dropout: float = 0.5,
        model_name: str = "custom_cnn",
        device: Optional[torch.device] = None
    ):
        """Initialize custom CNN.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            input_size: Input image size
            conv_layers: List of conv layer configs [{'out_channels': 64, 'kernel_size': 3, ...}]
            fc_layers: List of FC layer sizes [512, 256]
            dropout: Dropout probability
            model_name: Name of the model
            device: Device to run model on
        """
        super().__init__(num_classes, model_name, device)
        
        # Default architecture if not specified
        if conv_layers is None:
            conv_layers = [
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ]
        
        if fc_layers is None:
            fc_layers = [512]
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.dropout = dropout
        
        # Build convolutional layers
        self.conv_layers = self._build_conv_layers(conv_layers)
        
        # Calculate feature size
        self.feature_size = self._get_conv_output_size()
        
        # Build classifier
        self.classifier = self._build_classifier(fc_layers)
    
    def _build_conv_layers(self, conv_configs: List[Dict[str, Any]]) -> nn.Sequential:
        """Build convolutional layers from configs."""
        layers = []
        in_channels = self.input_channels
        
        for i, config in enumerate(conv_configs):
            out_channels = config['out_channels']
            kernel_size = config.get('kernel_size', 3)
            stride = config.get('stride', 1)
            padding = config.get('padding', 1)
            
            # Convolution
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            
            # Batch normalization
            layers.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Pooling (add after each conv block)
            layers.append(nn.MaxPool2d(2, 2))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self, fc_sizes: List[int]) -> nn.Sequential:
        """Build classifier from FC layer sizes."""
        layers = []
        in_features = self.feature_size
        
        for size in fc_sizes:
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.ReLU(inplace=True))
            in_features = size
        
        # Final classification layer
        layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(in_features, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def _get_conv_output_size(self) -> int:
        """Calculate output size of conv layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Convolutional features
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


# Registry of CNN models
CNN_MODELS = {
    "basic_cnn": BasicCNN,
    "resnet18": lambda **kwargs: ResNetWrapper(architecture="resnet18", **kwargs),
    "resnet34": lambda **kwargs: ResNetWrapper(architecture="resnet34", **kwargs),
    "resnet50": lambda **kwargs: ResNetWrapper(architecture="resnet50", **kwargs),
    "resnet101": lambda **kwargs: ResNetWrapper(architecture="resnet101", **kwargs),
    "efficientnet_b0": lambda **kwargs: EfficientNetWrapper(architecture="efficientnet_b0", **kwargs),
    "efficientnet_b1": lambda **kwargs: EfficientNetWrapper(architecture="efficientnet_b1", **kwargs),
    "custom_cnn": CustomCNN,
}


def get_cnn_model(model_name: str, num_classes: int, **kwargs) -> BaseModel:
    """Get a CNN model by name.
    
    Args:
        model_name: Name of the CNN model
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        CNN model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in CNN_MODELS:
        raise ValueError(
            f"Unknown CNN model: {model_name}. "
            f"Available models: {list(CNN_MODELS.keys())}"
        )
    
    return CNN_MODELS[model_name](num_classes=num_classes, **kwargs)