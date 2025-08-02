#!/usr/bin/env python3
"""Test script for model loading functionality."""

import sys
import os
from pathlib import Path

# Add MCal to path
mcal_root = Path(__file__).parent
sys.path.insert(0, str(mcal_root))

try:
    from configs import check_model_availability, get_model_path, get_dataset_config
    from src.models import ModelLoader, analyze_saved_models
    
    print("MCal Model Loading Test")
    print("=" * 50)
    
    # Check model availability
    print("\n1. Checking model availability:")
    check_model_availability()
    
    # Analyze saved models
    print("\n2. Analyzing saved models:")
    analysis = analyze_saved_models()
    
    for dataset, augs in analysis.items():
        if not augs:
            continue
        print(f"\n{dataset.upper()}:")
        for aug, info in augs.items():
            if 'error' in info:
                print(f"  {aug:15} ❌ {info['error']}")
            else:
                num_params = info.get('num_parameters', 'Unknown')
                num_classes = info.get('num_classes', 'Unknown')
                file_size = info.get('file_size_mb', 0)
                model_type = info.get('model_type', 'Unknown')
                print(f"  {aug:15} ✅ {num_params:,} params, {num_classes} classes, {file_size:.1f}MB ({model_type})")
    
    # Test loading a specific model
    print("\n3. Testing model loading:")
    try:
        loader = ModelLoader()
        
        # Try to load MRI vanilla model
        print("  Loading MRI vanilla model...")
        model = loader.load_dataset_model('mri', 'vanilla', model_type='timm_vit')
        print(f"  ✅ Successfully loaded MRI model: {type(model).__name__}")
        
        # Try with expectation tracking
        print("  Loading MRI model with expectation tracking...")
        tracked_model = loader.load_dataset_model(
            'mri', 'vanilla', 
            model_type='timm_vit',
            wrap_with_expectation_tracking=True
        )
        print(f"  ✅ Successfully loaded tracked model: {type(tracked_model).__name__}")
        
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
    
    print("\n4. Configuration test:")
    try:
        config = get_dataset_config('mri')
        print(f"  MRI config: {config['num_classes']} classes, {config['image_size']}px images")
        
        model_path = get_model_path('mri', 'vanilla')
        print(f"  MRI model path: {model_path}")
        print(f"  Model exists: {'✅' if model_path.exists() else '❌'}")
        
    except Exception as e:
        print(f"  ❌ Config error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("  pip install torch torchvision timm")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()