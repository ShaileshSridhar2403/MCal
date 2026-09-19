#!/usr/bin/env python3
"""
Download MRI Brain Tumor Dataset from Kaggle

This script downloads the Brain Tumor MRI dataset from Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Requirements:
1. Kaggle account
2. Kaggle API credentials (kaggle.json)
3. kaggle package installed (pip install kaggle)
"""

import sys
import shutil
import subprocess
from pathlib import Path

def check_kaggle_installed():
    """Check if kaggle CLI is installed."""
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        print(f"✓ Kaggle CLI found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("✗ Kaggle CLI not found")
        print("\nTo install: pip install kaggle")
        return False

def check_kaggle_credentials():
    """Check if kaggle credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"✓ Kaggle credentials found at: {kaggle_json}")
        return True
    else:
        print(f"✗ Kaggle credentials not found at: {kaggle_json}")
        print("\nTo setup:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_mri_dataset(output_dir: Path):
    """Download and extract MRI dataset."""
    print(f"\nDownloading to: {output_dir.absolute()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download using Kaggle API
    dataset_name = "masoudnickparvar/brain-tumor-mri-dataset"
    print(f"\nDownloading dataset: {dataset_name}")
    print("This may take a few minutes...")

    cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Download complete!")

        # Extract the zip file
        zip_path = output_dir / "brain-tumor-mri-dataset.zip"
        if zip_path.exists():
            print(f"\nExtracting {zip_path.name}...")
            shutil.unpack_archive(str(zip_path), str(output_dir))
            print("✓ Extraction complete!")

            # Check extracted structure
            print("\nDataset structure:")
            for item in sorted(output_dir.iterdir()):
                if item.is_dir():
                    n_subdirs = len(list(item.iterdir()))
                    print(f"  {item.name}/ ({n_subdirs} items)")

            # Clean up zip file
            print(f"\nCleaning up {zip_path.name}...")
            zip_path.unlink()
            print("✓ Cleanup complete!")

            return True
        else:
            print(f"✗ Error: Downloaded zip file not found at {zip_path}")
            return False
    else:
        print(f"✗ Download failed!")
        print(f"Error: {result.stderr}")
        return False

def verify_dataset_structure(data_dir: Path):
    """Verify the dataset has the expected structure."""
    print("\n" + "="*70)
    print("VERIFYING DATASET STRUCTURE")
    print("="*70)

    expected_dirs = ["Training", "Testing"]
    expected_classes = ["glioma", "meningioma", "notumor", "pituitary"]

    all_good = True

    for split in expected_dirs:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"✗ Missing directory: {split}")
            all_good = False
            continue

        print(f"\n{split}/")
        for class_name in expected_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"  ✗ Missing class: {class_name}")
                all_good = False
            else:
                n_images = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                print(f"  ✓ {class_name}: {n_images} images")

    if all_good:
        print("\n" + "="*70)
        print("✓ Dataset structure verified!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ Dataset structure incomplete")
        print("="*70)

    return all_good

def main():
    print("="*70)
    print("MRI BRAIN TUMOR DATASET DOWNLOADER")
    print("="*70)

    # Check prerequisites
    print("\n1. Checking prerequisites...")
    if not check_kaggle_installed():
        sys.exit(1)

    if not check_kaggle_credentials():
        sys.exit(1)

    # Set output directory
    output_dir = Path(__file__).resolve().parent / "experiments" / "vision" / "data"
    print(f"\n2. Dataset will be downloaded to: {output_dir.absolute()}")

    # Check if already exists
    if (output_dir / "Training").exists() and (output_dir / "Testing").exists():
        print("\n✓ Dataset already exists!")
        verify_dataset_structure(output_dir)
        print("\nIf you want to re-download, delete the existing data directory first:")
        print(f"  rm -rf {output_dir}")
        return

    # Confirm download
    print("\nDataset info:")
    print("  Name: Brain Tumor MRI Classification")
    print("  Size: ~400 MB")
    print("  Classes: 4 (glioma, meningioma, notumor, pituitary)")
    print("  Images: ~7000 total")

    response = input("\nProceed with download? (y/n): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return

    # Download and extract
    print("\n3. Downloading dataset...")
    if download_mri_dataset(output_dir):
        print("\n4. Verifying dataset structure...")
        verify_dataset_structure(output_dir)

        print("\n" + "="*70)
        print("✅ MRI DATASET READY!")
        print("="*70)
        print("\nYou can now run:")
        print("  cd experiments")
        print("  python run_faithfulness_experiments.py")
    else:
        print("\n✗ Download failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
