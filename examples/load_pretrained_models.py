#!/usr/bin/env python3
"""
Example: Loading and Using Pre-trained Models in MCal

This example demonstrates how to load the pre-trained models from XAI_Benchmark
and use them for inference and calibration in the MCal framework.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add MCal to path
mcal_root = Path(__file__).parent.parent
sys.path.insert(0, str(mcal_root))
sys.path.insert(0, str(mcal_root / "configs"))

from configs.model_dict import get_model_path, MODEL_DICT
from configs.dataset_configs import get_dataset_config, get_combined_config
import timm


class PretrainedModelDemo:
    """Demo class for working with pre-trained models."""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_model(self, dataset: str, augmentation: str = "vanilla"):
        """Load a pre-trained model for the given dataset and augmentation."""
        print(f"\nLoading {dataset} model with {augmentation} augmentation...")
        
        # Get configuration
        config = get_dataset_config(dataset)
        model_path = get_model_path(dataset, augmentation)
        
        print(f"  Dataset: {dataset}")
        print(f"  Classes: {config['num_classes']}")
        print(f"  Image size: {config['image_size']}")
        print(f"  Model file: {model_path.name}")
        
        # Create model
        model = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=False, 
            num_classes=config['num_classes']
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        print(f"  ✅ Model loaded successfully!")
        return model, config
    
    def demonstrate_inference(self, model, config):
        """Demonstrate model inference with dummy data."""
        print(f"  Testing inference...")
        
        # Create dummy input
        batch_size = 4
        img_size = config['image_size']
        dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(dummy_input)
            probabilities = torch.softmax(logits, dim=1)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Sample probabilities: {probabilities[0][:3].cpu().numpy()}")
        
        return probabilities
    
    def compare_models(self, dataset: str):
        """Compare vanilla and augmented models for a dataset."""
        print(f"\n{'='*60}")
        print(f"COMPARING MODELS FOR {dataset.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Load both vanilla and augmented models if available
        available_augs = list(MODEL_DICT.get(dataset, {}).keys())
        
        for aug in available_augs:
            try:
                model, config = self.load_model(dataset, aug)
                probs = self.demonstrate_inference(model, config)
                results[aug] = {
                    'model': model,
                    'config': config,
                    'sample_probs': probs[0].cpu().numpy()
                }
            except Exception as e:
                print(f"  ❌ Failed to load {aug}: {e}")
        
        # Compare probability distributions
        if len(results) > 1:
            print(f"\n  Comparing probability distributions:")
            print(f"  {'Model':<15} {'Max Prob':<10} {'Entropy':<10} {'Prediction'}")
            print(f"  {'-'*50}")
            
            for aug, result in results.items():
                probs = result['sample_probs']
                max_prob = float(np.max(probs))
                entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
                prediction = int(np.argmax(probs))
                
                print(f"  {aug:<15} {max_prob:<10.3f} {entropy:<10.3f} {prediction}")
        
        return results
    
    def demonstrate_calibration_setup(self, dataset: str):
        """Show how to set up models for calibration."""
        print(f"\n{'='*60}")
        print(f"CALIBRATION SETUP FOR {dataset.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load vanilla model (clean predictions)
            clean_model, config = self.load_model(dataset, 'vanilla')
            
            # Load augmented model if available (ablated predictions)
            aug_types = [aug for aug in MODEL_DICT.get(dataset, {}).keys() if aug != 'vanilla']
            if aug_types:
                ablated_model, _ = self.load_model(dataset, aug_types[0])
                
                print(f"\n  Calibration setup ready:")
                print(f"  - Clean model: vanilla")
                print(f"  - Ablated model: {aug_types[0]}")
                print(f"  - Classes: {config['num_classes']}")
                
                # Generate sample predictions for calibration
                dummy_input = torch.randn(10, 3, config['image_size'], config['image_size']).to(self.device)
                
                with torch.no_grad():
                    clean_probs = torch.softmax(clean_model(dummy_input), dim=1)
                    ablated_probs = torch.softmax(ablated_model(dummy_input), dim=1)
                
                print(f"  - Sample data: {dummy_input.shape}")
                print(f"  - Clean predictions: {clean_probs.shape}")
                print(f"  - Ablated predictions: {ablated_probs.shape}")
                
                # Show how this would be used with MCal calibrator
                print(f"\n  Example MCal usage:")
                print(f"  ```python")
                print(f"  from src.calibrators import MCal")
                print(f"  ")
                print(f"  calibrator = MCal(num_classes={config['num_classes']})")
                print(f"  calibrator.fit(ablated_probs, clean_probs)")
                print(f"  calibrated_probs = calibrator(ablated_probs)")
                print(f"  ```")
                
                return clean_probs, ablated_probs
            else:
                print(f"  ⚠️  No augmented model available for {dataset}")
                
        except Exception as e:
            print(f"  ❌ Error setting up calibration: {e}")
        
        return None, None


def main():
    """Main demonstration function."""
    print("MCal Pre-trained Model Demo")
    print("="*60)
    
    demo = PretrainedModelDemo()
    
    # Available datasets with models
    datasets_to_test = ['mri', 'breakhis', 'chexpert', 'imagenette']
    
    for dataset in datasets_to_test:
        if dataset in MODEL_DICT:
            try:
                # Compare models for this dataset
                results = demo.compare_models(dataset)
                
                # Show calibration setup
                clean_probs, ablated_probs = demo.demonstrate_calibration_setup(dataset)
                
            except Exception as e:
                print(f"❌ Error with {dataset}: {e}")
    
    print(f"\n{'='*60}")
    print("Demo completed!")
    print("\nTo use these models in your own code:")
    print("1. Import MCal: from src.models import load_model_for_dataset")
    print("2. Load model: model = load_model_for_dataset('mri', 'vanilla')")
    print("3. Use for inference or calibration")
    print("\nAll pre-trained models from XAI_Benchmark are now available in MCal!")


if __name__ == "__main__":
    main()