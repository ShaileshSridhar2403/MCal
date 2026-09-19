"""Utility functions for MCal."""

from .optimization import (
    get_expectation,
    make_one_hot,
    kl_divergence,
    apply_lambda_adjustment,
    find_optimal_lambda_batch
)
from .io import save_results, load_results
from .visualization import plot_kl_divergence

__all__ = [
    "get_expectation",
    "make_one_hot",
    "kl_divergence",
    "apply_lambda_adjustment",
    "find_optimal_lambda_batch",
    "save_results",
    "load_results",
    "plot_kl_divergence"
]
