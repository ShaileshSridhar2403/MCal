#!/usr/bin/env python3
"""
Quick script to process PhysioNet data using the exact XAI_Benchmark pipeline.

This script looks for PhysionetChallenge2012-set-a.csv.gz in common locations
and processes it into the required missingness-level format.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from process_physionet_data import process_physionet_data


def find_physionet_file():
    """Find PhysioNet data file in common locations."""
    possible_paths = [
        # Current directory
        "./PhysionetChallenge2012-set-a.csv.gz",
        # Data directory
        "../../../data/PhysionetChallenge2012-set-a.csv.gz",
        # XAI_Benchmark directory
        "../../../../XAI-Benchmark/PhysionetChallenge2012-set-a.csv.gz",
        # Home directory
        "~/PhysionetChallenge2012-set-a.csv.gz",
        # Downloads
        "~/Downloads/PhysionetChallenge2012-set-a.csv.gz"
    ]

    for path in possible_paths:
        expanded_path = Path(path).expanduser().resolve()
        if expanded_path.exists():
            print(f"Found PhysioNet data at: {expanded_path}")
            return str(expanded_path)

    return None


def main():
    """Main execution."""
    print("=" * 60)
    print("PhysioNet Data Processing for MCal Tabular Benchmarks")
    print("=" * 60)

    # Try to find the PhysioNet file
    physionet_file = find_physionet_file()

    if physionet_file is None:
        print("❌ PhysioNet data file not found!")
        print("\nPlease ensure PhysionetChallenge2012-set-a.csv.gz is available in one of these locations:")
        print("  - Current directory")
        print("  - MCal/data/ directory")
        print("  - XAI_Benchmark/ directory")
        print("  - Home directory")
        print("  - Downloads directory")
        print("\nOr run with explicit path:")
        print("  python process_physionet_data.py --input_path /path/to/PhysionetChallenge2012-set-a.csv.gz")
        return False

    # Process the data
    output_dir = "/home/antonxue/shailesh/MCal/data/tabular/missingness_levels"
    print(f"Processing: {physionet_file}")
    print(f"Output directory: {output_dir}")

    try:
        X = process_physionet_data(
            input_path=physionet_file,
            output_dir=output_dir,
            from_txt=False
        )

        print(f"\n🎉 Success! Generated missingness-level files:")

        # List generated files
        missingness_dir = Path(output_dir)
        if missingness_dir.exists():
            files = sorted(list(missingness_dir.glob("*.csv.gz")))
            for f in files:
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  - {f.name} ({size_mb:.1f} MB)")

        print(f"\nNow you can run the tabular benchmark:")
        print(f"  python tabular_kl_benchmark.py --samples 100 --runs 1")

        return True

    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)