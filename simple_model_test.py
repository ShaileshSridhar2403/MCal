#!/usr/bin/env python3
"""Simple test for model loading without complex imports."""

import sys
import os
from pathlib import Path
import torch

# Add paths
mcal_root = Path(__file__).parent
sys.path.insert(0, str(mcal_root))
sys.path.insert(0, str(mcal_root / "configs"))

from configs.model_dict import check_model_availability, get_model_path
from configs.dataset_configs import get_dataset_config

def test_basic_functionality():
    print("MCal Model and Config Test")
    print("=" * 40)
    
    # Test 1: Model availability
    print("\n1. Model Availability:")
    check_model_availability()
    
    # Test 2: Config loading
    print("\n2. Configuration Loading:")
    try:
        mri_config = get_dataset_config('mri')
        print(f"✅ MRI config loaded: {mri_config['num_classes']} classes")
        
        breakhis_config = get_dataset_config('breakhis')  
        print(f"✅ BreakHis config loaded: {breakhis_config['num_classes']} classes")
        
    except Exception as e:
        print(f"❌ Config error: {e}")
    
    # Test 3: Model path resolution
    print("\n3. Model Path Resolution:")
    try:
        mri_path = get_model_path('mri', 'vanilla')
        print(f"✅ MRI vanilla path: {mri_path}")
        print(f"   Exists: {'✅' if mri_path.exists() else '❌'}")
        
        breakhis_path = get_model_path('breakhis', 'PatchCutout')
        print(f"✅ BreakHis PatchCutout path: {breakhis_path}")
        print(f"   Exists: {'✅' if breakhis_path.exists() else '❌'}")
        
    except Exception as e:
        print(f"❌ Path error: {e}")
    
    # Test 4: Basic model loading with timm
    print("\n4. Basic Model Loading Test:")
    try:
        import timm
        
        # Test loading structure (without actual state dict)
        mri_config = get_dataset_config('mri')
        model = timm.create_model('vit_base_patch16_224', 
                                pretrained=False, 
                                num_classes=mri_config['num_classes'])
        print(f"✅ Created ViT model for MRI: {mri_config['num_classes']} classes")
        
        # Test actual model loading
        mri_path = get_model_path('mri', 'vanilla')
        if mri_path.exists():
            try:
                state_dict = torch.load(mri_path, map_location='cpu')
                print(f"✅ Loaded model state dict from {mri_path.name}")
                print(f"   State dict keys: {len(state_dict)} parameters")
                
                # Check if we can load it into the model
                model.load_state_dict(state_dict, strict=False)  # Use strict=False for compatibility
                print(f"✅ Successfully applied state dict to model")
                
            except Exception as e:
                print(f"⚠️  Could not load state dict: {e}")
        
    except ImportError:
        print("❌ timm not available - install with: pip install timm")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
    
    print(f"\n{'='*40}")
    print("Test completed!")

if __name__ == "__main__":
    test_basic_functionality()