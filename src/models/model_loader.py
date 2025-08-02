"""Model loading utilities for pre-trained models."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
import logging

from .vision import VisionTransformer, get_vision_model
from .base_model import BaseModel, ExpectationTrackingModel
try:
    from ...configs import get_model_path, get_dataset_config, get_vit_config
except ImportError:
    import sys
    from pathlib import Path
    # Add configs to path
    config_path = Path(__file__).parent.parent.parent / "configs"
    sys.path.insert(0, str(config_path))
    from model_dict import get_model_path
    from dataset_configs import get_dataset_config, get_vit_config

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading pre-trained models."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize model loader.
        
        Args:
            device: Device to load models on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_timm_vit(
        self,
        model_path: Union[str, Path],
        num_classes: int,
        model_name: str = 'vit_base_patch16_224',
        strict: bool = True
    ) -> nn.Module:
        """Load a TIMM Vision Transformer model.
        
        Args:
            model_path: Path to model checkpoint
            num_classes: Number of output classes
            model_name: TIMM model name
            strict: Whether to strictly enforce state dict keys
            
        Returns:
            Loaded model
        """
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for loading TIMM models. Install with: pip install timm")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading TIMM model from: {model_path}")
        
        # Create model
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # Load state dict
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=strict)
        except Exception as e:
            logger.error(f"Error loading model state dict: {e}")
            if strict:
                logger.info("Retrying with strict=False...")
                model.load_state_dict(state_dict, strict=False)
            else:
                raise
        
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Successfully loaded model with {num_classes} classes")
        return model
    
    def load_mcal_vit(
        self,
        model_path: Union[str, Path],
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        strict: bool = True
    ) -> VisionTransformer:
        """Load an MCal Vision Transformer model.
        
        Args:
            model_path: Path to model checkpoint
            num_classes: Number of output classes
            img_size: Input image size
            patch_size: Patch size
            embed_dim: Embedding dimension
            depth: Number of layers
            num_heads: Number of attention heads
            strict: Whether to strictly enforce state dict keys
            
        Returns:
            Loaded VisionTransformer model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading MCal ViT model from: {model_path}")
        
        # Create model
        model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            device=self.device
        )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            model_state_dict = state_dict['model_state_dict']
        else:
            model_state_dict = state_dict
        
        model.load_state_dict(model_state_dict, strict=strict)
        model.eval()
        
        logger.info(f"Successfully loaded MCal ViT model with {num_classes} classes")
        return model
    
    def load_dataset_model(
        self,
        dataset: str,
        augmentation: str = "vanilla",
        model_type: str = "timm_vit",
        wrap_with_expectation_tracking: bool = False,
        **kwargs
    ) -> Union[nn.Module, BaseModel]:
        """Load a pre-trained model for a specific dataset.
        
        Args:
            dataset: Dataset name (e.g., 'mri', 'breakhis')
            augmentation: Augmentation type (e.g., 'vanilla', 'PatchCutout')
            model_type: Type of model to load ('timm_vit', 'mcal_vit')
            wrap_with_expectation_tracking: Whether to wrap with ExpectationTrackingModel
            **kwargs: Additional arguments for model creation
            
        Returns:
            Loaded model
        """
        # Get model path
        model_path = get_model_path(dataset, augmentation)
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset)
        num_classes = dataset_config['num_classes']
        img_size = dataset_config.get('image_size', 224)
        patch_size = dataset_config.get('patch_size', 16)
        
        # Load model based on type
        if model_type == "timm_vit":
            # Use TIMM ViT (default for compatibility with existing models)
            model = self.load_timm_vit(
                model_path=model_path,
                num_classes=num_classes,
                **kwargs
            )
        elif model_type == "mcal_vit":
            # Use MCal ViT implementation
            vit_config = get_vit_config(kwargs.get('vit_variant', 'base'))
            model = self.load_mcal_vit(
                model_path=model_path,
                num_classes=num_classes,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=vit_config['embed_dim'],
                depth=vit_config['depth'],
                num_heads=vit_config['num_heads'],
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Wrap with expectation tracking if requested
        if wrap_with_expectation_tracking:
            model_name = f"{dataset}_{augmentation}_{model_type}"
            model = ExpectationTrackingModel(
                base_model=model,
                num_classes=num_classes,
                model_name=model_name,
                device=self.device
            )
        
        return model
    
    def analyze_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze a model checkpoint to extract information.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Dictionary with model information
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'file_path': str(model_path),
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
        }
        
        # Analyze checkpoint structure
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # MCal format
                info['format'] = 'mcal'
                info['keys'] = list(checkpoint.keys())
                if 'model_info' in checkpoint:
                    info['model_info'] = checkpoint['model_info']
                state_dict = checkpoint['model_state_dict']
            else:
                # Direct state dict format (TIMM/PyTorch)
                info['format'] = 'direct_state_dict'
                state_dict = checkpoint
        else:
            info['format'] = 'unknown'
            return info
        
        # Analyze state dict
        info['num_parameters'] = sum(p.numel() for p in state_dict.values())
        info['parameter_shapes'] = {k: list(v.shape) for k, v in state_dict.items()}
        
        # Try to infer model architecture
        if 'head.weight' in state_dict:
            info['num_classes'] = state_dict['head.weight'].shape[0]
            info['embed_dim'] = state_dict['head.weight'].shape[1]
        elif 'classifier.weight' in state_dict:
            info['num_classes'] = state_dict['classifier.weight'].shape[0]
        
        # Check for ViT-specific parameters
        if 'pos_embed' in state_dict:
            info['model_type'] = 'vision_transformer'
            pos_embed_shape = state_dict['pos_embed'].shape
            info['pos_embed_shape'] = list(pos_embed_shape)
            # Estimate patch size and image size
            num_patches = pos_embed_shape[1] - 1  # Subtract class token
            info['estimated_num_patches'] = num_patches
        
        return info


# Convenience functions
def load_model_for_dataset(
    dataset: str,
    augmentation: str = "vanilla",
    device: Optional[torch.device] = None,
    **kwargs
) -> nn.Module:
    """Convenience function to load a model for a dataset.
    
    Args:
        dataset: Dataset name
        augmentation: Augmentation type
        device: Device to load on
        **kwargs: Additional arguments
        
    Returns:
        Loaded model
    """
    loader = ModelLoader(device=device)
    return loader.load_dataset_model(dataset, augmentation, **kwargs)


def load_expectation_tracking_model(
    dataset: str,
    augmentation: str = "vanilla",
    device: Optional[torch.device] = None,
    **kwargs
) -> ExpectationTrackingModel:
    """Load a model wrapped with expectation tracking.
    
    Args:
        dataset: Dataset name
        augmentation: Augmentation type
        device: Device to load on
        **kwargs: Additional arguments
        
    Returns:
        ExpectationTrackingModel instance
    """
    loader = ModelLoader(device=device)
    return loader.load_dataset_model(
        dataset, 
        augmentation, 
        wrap_with_expectation_tracking=True,
        **kwargs
    )


def analyze_saved_models() -> Dict[str, Dict[str, Any]]:
    """Analyze all saved models and return information.
    
    Returns:
        Dictionary mapping model names to their information
    """
    from ..configs import list_available_models
    
    loader = ModelLoader()
    analysis = {}
    
    available_models = list_available_models()
    
    for dataset, augs in available_models.items():
        analysis[dataset] = {}
        for aug, info in augs.items():
            if info['exists']:
                try:
                    model_analysis = loader.analyze_model(info['path'])
                    analysis[dataset][aug] = model_analysis
                except Exception as e:
                    analysis[dataset][aug] = {'error': str(e)}
            else:
                analysis[dataset][aug] = {'error': 'Model file not found'}
    
    return analysis


if __name__ == "__main__":
    # Example usage
    print("Analyzing saved models...")
    analysis = analyze_saved_models()
    
    for dataset, augs in analysis.items():
        print(f"\n{dataset.upper()}:")
        for aug, info in augs.items():
            if 'error' in info:
                print(f"  {aug}: ERROR - {info['error']}")
            else:
                num_params = info.get('num_parameters', 'Unknown')
                num_classes = info.get('num_classes', 'Unknown')
                file_size = info.get('file_size_mb', 0)
                print(f"  {aug}: {num_params:,} params, {num_classes} classes, {file_size:.1f}MB")