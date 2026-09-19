"""Run the faithfulness, deletion and insertion experiments for one dataset.

usage: python -m experiments.run_faithfulness --dataset DATASET

Each dataset's experiment lives in experiments/faithfulness/<dataset>.py.
Results are written to experiments/results/.
"""

import argparse
import runpy

DATASETS = ["mri", "breakhis", "chexpert", "ctg", "physionet", "breast_cancer"]


def main():
    parser = argparse.ArgumentParser(
        description="Run the faithfulness, deletion and insertion experiments for one dataset.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    args = parser.parse_args()
    runpy.run_module(f"experiments.faithfulness.{args.dataset}", run_name="__main__")


if __name__ == "__main__":
    main()
