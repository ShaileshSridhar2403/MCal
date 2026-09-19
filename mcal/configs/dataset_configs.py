"""Dataset-specific configurations for model training and evaluation."""

from typing import Dict, Any


# Base configuration template
BASE_CONFIG = {
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'batch_size': 32,
    'num_epochs': 10,
    'drop_rate': 0.0,
    'batch_train_loss_freq': 10,
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'warmup_epochs': 1,
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    'breakhis': {
        **BASE_CONFIG,
        'num_classes': 8,
        'learning_rate': 0.00001,
        'weight_decay': 0.00001,
        'batch_size': 32,
        'num_epochs': 10,
        'image_size': 224,
        'patch_size': 56,  # Based on model names
        'input_shape': (224, 224, 3),
    },
    
    'mri': {
        **BASE_CONFIG,
        'num_classes': 4,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 256,
        'num_epochs': 35,  # Based on model name "35e"
        'image_size': 128,  # Based on vit_model_config.py
        'patch_size': 56,   # Based on model names "ps64"
        'input_shape': (128, 128, 3),
    },
    
    'chexpert': {
        **BASE_CONFIG,
        'num_classes': 2,  # Binary classification (based on actual model predictions)
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'num_epochs': 20,
        'image_size': 224,
        'patch_size': 56,   # Based on model names "ps64"
        'input_shape': (224, 224, 3),
    },
    
    'imagenette': {
        **BASE_CONFIG,
        'num_classes': 10,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'num_epochs': 70,   # Based on model name "70e"
        'image_size': 224,
        'patch_size': 64,   # Based on model names "ps64"
        'input_shape': (224, 224, 3),
    },
    
    'imagenet': {
        **BASE_CONFIG,
        'num_classes': 1000,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 256,
        'num_epochs': 100,
        'image_size': 224,
        'patch_size': 16,   # Standard ViT patch size
        'input_shape': (224, 224, 3),
    }
}

# Vision Transformer specific configurations
VIT_CONFIGS = {
    'base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    },
    
    'small': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    },
    
    'large': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    },
    
    # Custom config based on original vit_model_config.py
    'custom_small': {
        'embed_dim': 64,        # projection_dim
        'depth': 8,             # transformer_layers
        'num_heads': 4,         # num_heads
        'mlp_ratio': 2.0,       # Based on transformer_units
        'dropout': 0.1,
        'mlp_head_units': [2048, 1024],  # From original config
    }
}

# Training configurations for different scenarios
TRAINING_CONFIGS = {
    'vanilla': {
        'augmentation': None,
        'description': 'Standard training without augmentation'
    },
    
    'PatchCutout': {
        'augmentation': 'PatchCutout',
        'patch_drop_ratio': 0.5,
        'random_removal': True,
        'random_dist': 'binomial',
        'description': 'Training with PatchCutout augmentation'
    },
    
    'Cutout': {
        'augmentation': 'Cutout',
        'n_holes': 5,
        'length': 32,
        'description': 'Training with standard Cutout augmentation'
    }
}


def get_dataset_config(dataset: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Configuration dictionary
        
    Raises:
        KeyError: If dataset not found
    """
    if dataset not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset].copy()


def get_vit_config(variant: str = 'base') -> Dict[str, Any]:
    """Get Vision Transformer configuration.
    
    Args:
        variant: ViT variant ('base', 'small', 'large', 'custom_small')
        
    Returns:
        ViT configuration dictionary
    """
    if variant not in VIT_CONFIGS:
        raise KeyError(f"Unknown ViT variant: {variant}. Available: {list(VIT_CONFIGS.keys())}")
    
    return VIT_CONFIGS[variant].copy()


def get_training_config(training_type: str = 'vanilla') -> Dict[str, Any]:
    """Get training configuration.
    
    Args:
        training_type: Training type ('vanilla', 'PatchCutout', 'Cutout')
        
    Returns:
        Training configuration dictionary
    """
    if training_type not in TRAINING_CONFIGS:
        raise KeyError(f"Unknown training type: {training_type}. Available: {list(TRAINING_CONFIGS.keys())}")
    
    return TRAINING_CONFIGS[training_type].copy()


def get_combined_config(dataset: str, vit_variant: str = 'base', training_type: str = 'vanilla') -> Dict[str, Any]:
    """Get combined configuration for dataset, model, and training.
    
    Args:
        dataset: Dataset name
        vit_variant: ViT variant
        training_type: Training type
        
    Returns:
        Combined configuration dictionary
    """
    config = {}
    
    # Base dataset config
    config.update(get_dataset_config(dataset))
    
    # ViT model config
    config['model'] = get_vit_config(vit_variant)
    
    # Training config
    config['training'] = get_training_config(training_type)
    
    return config


# Legacy aliases for compatibility
breakhis_config = DATASET_CONFIGS['breakhis']
mri_config = DATASET_CONFIGS['mri']


def print_config_summary():
    """Print summary of all available configurations."""
    print("Dataset Configurations:")
    print("=" * 50)
    
    for dataset, config in DATASET_CONFIGS.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Classes: {config['num_classes']}")
        print(f"  Image Size: {config['image_size']}")
        print(f"  Patch Size: {config['patch_size']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Epochs: {config['num_epochs']}")
    
    print(f"\nViT Variants: {list(VIT_CONFIGS.keys())}")
    print(f"Training Types: {list(TRAINING_CONFIGS.keys())}")


if __name__ == "__main__":
    print_config_summary()