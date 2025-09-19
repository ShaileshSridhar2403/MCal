#!/usr/bin/env python3
"""
Diagnose LLaMA Model Files

Check which safetensors file is causing the corruption issue.
"""

import sys
from pathlib import Path

def check_safetensors_files():
    """Check each safetensors file for corruption."""

    model_path = Path("~/shailesh/MCal/saved_models/language/Meta-Llama-3-8B-Instruct/").expanduser()
    print(f"Checking model files in: {model_path}")

    # Find all safetensors files
    safetensors_files = list(model_path.glob("*.safetensors"))
    print(f"Found {len(safetensors_files)} safetensors files:")

    for file_path in sorted(safetensors_files):
        print(f"\n--- Checking {file_path.name} ---")
        print(f"Size: {file_path.stat().st_size:,} bytes")

        try:
            # Try to import safetensors and check the file
            from safetensors import safe_open

            with safe_open(file_path, framework="pt") as f:
                keys = list(f.keys())
                print(f"✓ File readable - contains {len(keys)} tensors")
                if keys:
                    print(f"  Sample keys: {keys[:3]}...")

        except ImportError:
            print("⚠️  safetensors not available - trying basic file check")
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(100)
                    print(f"✓ File readable - header: {header[:50]}...")
            except Exception as e:
                print(f"✗ Error reading file: {e}")

        except Exception as e:
            print(f"✗ SafeTensors error: {e}")
            print(f"  This file appears to be corrupted!")

if __name__ == "__main__":
    print("LLaMA Model File Diagnostics")
    print("="*50)
    check_safetensors_files()