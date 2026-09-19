"""Data loaders for the vision datasets used in the paper.

The language and tabular experiments load their data with their own
scripts under ``experiments/``.
"""

from .base_loader import BaseDataLoader, VisionDataLoader
from .vision_loaders import (
    BreakHisLoader,
    MRILoader,
    ChexPertLoader,
    get_vision_loader,
    mri_full_setup,
    VISION_LOADERS
)


def get_loader(dataset_name: str, **kwargs):
    """Get a dataset loader by name.

    Args:
        dataset_name: Name of the dataset
        **kwargs: Arguments passed to loader constructor

    Returns:
        Dataset loader instance

    Raises:
        ValueError: If dataset_name is not recognized
    """
    return get_vision_loader(dataset_name, **kwargs)


def list_datasets() -> list:
    """List available datasets."""
    return list(VISION_LOADERS.keys())


__all__ = [
    # Base classes
    "BaseDataLoader",
    "VisionDataLoader",

    # Vision loaders
    "BreakHisLoader",
    "MRILoader",
    "ChexPertLoader",
    "get_vision_loader",
    "mri_full_setup",

    # Utilities
    "get_loader",
    "list_datasets",
    "VISION_LOADERS",
]
