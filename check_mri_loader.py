#!/usr/bin/env python3
"""
Test script for MRILoader.setup_dataset() function.

This script helps test the MRI data loading functionality as used in 
experiments/mri_mcal_linear_mlp_test.ipynb
"""

from pathlib import Path
import argparse
import logging
import torch


from mcal.data.loaders.vision_loaders import MRILoader
from mcal.paths import DATA_ROOT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mri_loader_basic(data_dir=None, image_size=224):
    """Test basic MRILoader functionality."""
    print("=" * 60)
    print("BASIC MRI LOADER TEST")
    print("=" * 60)
    
    # Initialize loader
    loader = MRILoader(data_dir=data_dir, image_size=image_size)
    
    print(f"✅ Initialized MRILoader:")
    print(f"   Dataset name: {loader.dataset_name}")
    print(f"   Number of classes: {loader.num_classes}")
    print(f"   Class names: {loader.class_names}")
    print(f"   Data directory: {loader.data_dir}")
    print(f"   Image size: {loader.image_size}")
    
    return loader

def test_setup_dataset(loader, download=False):
    """Test the setup_dataset() method."""
    print("\n" + "=" * 60)
    print("SETUP DATASET TEST")
    print("=" * 60)
    
    try:
        # Check if data exists
        train_dir = loader.data_dir / "Training"
        test_dir = loader.data_dir / "Testing"
        
        print(f"Checking for data directories:")
        print(f"   Training dir exists: {train_dir.exists()}")
        print(f"   Testing dir exists: {test_dir.exists()}")
        
        if not train_dir.exists() and not download:
            print("\n⚠️  Data not found. Run with --download to download dataset")
            print("   Note: This requires Kaggle API credentials")
            return None, None, None
        
        # Setup dataset
        print("\n🔄 Setting up dataset...")
        train_dataset, test_dataset, val_dataset = loader.setup_dataset()
        
        print("✅ Dataset setup complete!")
        
        # Print dataset information
        if train_dataset:
            print(f"   Training samples: {len(train_dataset)}")
            print(f"   Training classes: {len(train_dataset.classes)}")
            print(f"   Class names: {train_dataset.classes}")
        
        if test_dataset:
            print(f"   Test samples: {len(test_dataset)}")
            print(f"   Test classes: {len(test_dataset.classes)}")
        
        if val_dataset:
            print(f"   Validation samples: {len(val_dataset)}")
        
        return train_dataset, test_dataset, val_dataset
        
    except Exception as e:
        print(f"❌ Error setting up dataset: {e}")
        return None, None, None

def test_sample_batch(loader, dataset, dataset_name="dataset"):
    """Test loading a sample batch from the dataset."""
    if dataset is None:
        print(f"\n⏭️  Skipping batch test for {dataset_name} (dataset is None)")
        return
    
    print(f"\n" + "=" * 60)
    print(f"BATCH LOADING TEST ({dataset_name.upper()})")
    print("=" * 60)
    
    try:
        # Create dataloader
        dataloader = loader.get_dataloader(dataset, batch_size=4, shuffle=True)
        
        # Get first batch
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        
        print("✅ Successfully loaded batch!")
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Image dtype: {images.dtype}")
        print(f"   Label dtype: {labels.dtype}")
        print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   Sample labels: {labels.tolist()}")
        
        # Map labels to class names if available
        if hasattr(dataset, 'classes'):
            label_names = [dataset.classes[label] for label in labels]
            print(f"   Sample class names: {label_names}")
        
    except Exception as e:
        print(f"❌ Error loading batch: {e}")

def test_transforms(loader):
    """Test different transform configurations."""
    print("\n" + "=" * 60)
    print("TRANSFORMS TEST")
    print("=" * 60)
    
    # Test different splits
    splits = ["train", "test"]
    for split in splits:
        try:
            transform = loader.get_transforms(split=split)
            print(f"✅ {split.capitalize()} transforms: {len(transform.transforms)} steps")
            for i, t in enumerate(transform.transforms):
                print(f"   {i+1}. {type(t).__name__}")
        except Exception as e:
            print(f"❌ Error getting {split} transforms: {e}")
    
    # Test with augmentation
    try:
        aug_transform = loader.get_transforms(split="test", augmentation="PatchCutout")
        print(f"✅ Test transforms with PatchCutout: {len(aug_transform.transforms)} steps")
        for i, t in enumerate(aug_transform.transforms):
            print(f"   {i+1}. {type(t).__name__}")
    except Exception as e:
        print(f"⚠️  PatchCutout augmentation not available: {e}")

def test_dataset_info(loader):
    """Test dataset information retrieval."""
    print("\n" + "=" * 60)
    print("DATASET INFO TEST")
    print("=" * 60)
    
    try:
        info = loader.get_dataset_info()
        print("✅ Dataset information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Error getting dataset info: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test MRILoader.setup_dataset() functionality")
    parser.add_argument("--data-dir", type=str, help="Data directory path")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--download", action="store_true", help="Download dataset if not found")
    parser.add_argument("--test-batch", action="store_true", help="Test batch loading")
    parser.add_argument("--test-transforms", action="store_true", help="Test transforms")
    parser.add_argument("--test-info", action="store_true", help="Test dataset info")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = DATA_ROOT
    
    print(f"🧠 Testing MRILoader with:")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Image size: {args.image_size}")
    print(f"   Download if missing: {args.download}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    # Test basic loader functionality
    loader = test_mri_loader_basic(data_dir=args.data_dir, image_size=args.image_size)
    
    # Test setup_dataset
    train_dataset, test_dataset, val_dataset = test_setup_dataset(loader, download=args.download)
    
    # Run additional tests if requested
    if args.test_batch or args.all:
        test_sample_batch(loader, train_dataset, "training")
        test_sample_batch(loader, test_dataset, "testing")
    
    if args.test_transforms or args.all:
        test_transforms(loader)
    
    if args.test_info or args.all:
        test_dataset_info(loader)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if train_dataset or test_dataset:
        print("✅ MRILoader.setup_dataset() working correctly!")
        if train_dataset:
            print(f"   Training dataset: {len(train_dataset)} samples")
        if test_dataset:
            print(f"   Test dataset: {len(test_dataset)} samples")
    else:
        print("⚠️  Could not load datasets. Check data directory or use --download")
    
    print("\n📝 Usage examples:")
    print("   # Basic test")
    print("   python check_mri_loader.py")
    print()
    print("   # Download dataset and run all tests")
    print("   python check_mri_loader.py --download --all")
    print()
    print("   # Test with custom data directory")
    print("   python check_mri_loader.py --data-dir /path/to/data --test-batch")

if __name__ == "__main__":
    main()