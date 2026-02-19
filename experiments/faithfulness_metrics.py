#!/usr/bin/env python3
"""
Faithfulness and Deletion/Insertion Metrics for MCal Experiments

Implements additional metrics to evaluate explanation quality:
- Faithfulness (Pearson correlation)
- Deletion metric (iterative feature removal)
- Insertion metric (iterative feature addition)

These metrics complement the existing sufficiency and comprehensiveness metrics
to provide a more complete evaluation of explanation quality.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from scipy.stats import pearsonr
from abc import ABC, abstractmethod
import scipy.stats as stats



class BaseFaithfulnessMetric(ABC):
    """Base class for faithfulness evaluation metrics."""

    @abstractmethod
    def compute(self, *args, **kwargs) -> Union[float, Dict[str, float]]:
        """Compute the metric value."""
        pass


# =============================================================================
# Image Metrics
# =============================================================================

class ImageFaithfulnessPearson:
    """
    Faithfulness metric using Pearson correlation for image explanations.

    Measures the correlation between attribution scores and actual model
    output changes when features are ablated.
    """

    def __init__(self, patch_size: int = 56, image_size: int = 224,
                 baseline_value: float = 0.0):
        """
        Initialize Faithfulness Pearson metric.

        Args:
            patch_size: Size of each patch
            image_size: Size of the full image
            baseline_value: Value to use for ablated patches
        """
        self.patch_size = patch_size
        self.image_size = image_size
        self.baseline_value = baseline_value

    def create_patch_mask(self, patch_idx: int) -> torch.Tensor:
        """Create a binary mask for a single patch."""
        n_patches_per_dim = self.image_size // self.patch_size
        mask = torch.zeros((self.image_size, self.image_size))

        row = patch_idx // n_patches_per_dim
        col = patch_idx % n_patches_per_dim

        row_start = row * self.patch_size
        row_end = row_start + self.patch_size
        col_start = col * self.patch_size
        col_end = col_start + self.patch_size

        mask[row_start:row_end, col_start:col_end] = 1

        return mask

    def compute(self, model: nn.Module, image: torch.Tensor,
                attributions: torch.Tensor, true_label: int) -> float:
        """
        Compute faithfulness using Pearson correlation.

        Args:
            model: PyTorch model
            image: Input image (C, H, W)
            attributions: Feature importance scores (n_patches,)
            true_label: True class label

        Returns:
            Pearson correlation coefficient (higher is better, range [-1, 1])
        """
        device = image.device
        n_patches = len(attributions)

        # Get original prediction
        with torch.no_grad():
            pred_original = model(image.unsqueeze(0))[0, true_label].item()

        # For each patch, ablate it and measure prediction change
        prediction_changes = []
        attribution_scores = attributions.abs().cpu().numpy()

        for patch_idx in range(n_patches):
            # Create mask for this patch
            mask = self.create_patch_mask(patch_idx).to(device)

            # Ablate the patch
            ablated_image = image.clone()
            for c in range(image.shape[0]):
                ablated_image[c] = ablated_image[c] * (1 - mask) + self.baseline_value * mask

            # Get prediction on ablated image
            with torch.no_grad():
                pred_ablated = model(ablated_image.unsqueeze(0))[0, true_label].item()

            # Record the change in prediction
            prediction_changes.append(abs(pred_original - pred_ablated))

        # Compute Pearson correlation
        if len(prediction_changes) > 1:
            # corr, _ = pearsonr(attribution_scores, prediction_changes) #original version
            res = stats.weightedtau(attribution_scores, prediction_changes)
            return res.statistic if not np.isnan(res.statistic) else 0.0
        else:
            return 0.0


class ImageDeletionMetric:
    """
    Deletion metric for image explanations.

    Progressively removes features in order of importance and measures
    the area under the prediction score curve.
    """

    def __init__(self, patch_size: int = 56, image_size: int = 224,
                 baseline_value: float = 0.0, n_steps: int = 10):
        """
        Initialize Deletion metric.

        Args:
            patch_size: Size of each patch
            image_size: Size of the full image
            baseline_value: Value to use for deleted patches
            n_steps: Number of deletion steps
        """
        self.patch_size = patch_size
        self.image_size = image_size
        self.baseline_value = baseline_value
        self.n_steps = n_steps

    def create_mask_from_indices(self, indices: List[int]) -> torch.Tensor:
        """Create a binary mask from patch indices."""
        n_patches_per_dim = self.image_size // self.patch_size
        mask = torch.zeros((self.image_size, self.image_size))

        for idx in indices:
            row = idx // n_patches_per_dim
            col = idx % n_patches_per_dim

            row_start = row * self.patch_size
            row_end = row_start + self.patch_size
            col_start = col * self.patch_size
            col_end = col_start + self.patch_size

            mask[row_start:row_end, col_start:col_end] = 1

        return mask

    def compute(self, model: nn.Module, image: torch.Tensor,
                attributions: torch.Tensor, true_label: int) -> Dict[str, Any]:
        """
        Compute deletion metric.

        Args:
            model: PyTorch model
            image: Input image (C, H, W)
            attributions: Feature importance scores (n_patches,)
            true_label: True class label

        Returns:
            Dictionary with 'auc' (area under curve), 'scores' (prediction scores
            at each step), and 'fractions' (fraction of features deleted)
        """
        device = image.device
        n_patches = len(attributions)

        # Sort patches by importance (descending)
        sorted_indices = torch.argsort(torch.abs(attributions), descending=True).cpu().numpy()

        # Get original prediction
        with torch.no_grad():
            pred_original = torch.softmax(model(image.unsqueeze(0)), dim=1)[0, true_label].item()

        # Progressively delete patches
        scores = [pred_original]
        fractions = [0.0]

        patches_per_step = max(1, n_patches // self.n_steps)

        for step in range(1, self.n_steps + 1):
            # Determine which patches to delete
            n_patches_to_delete = min(step * patches_per_step, n_patches)
            patches_to_delete = sorted_indices[:n_patches_to_delete].tolist()

            # Create mask
            mask = self.create_mask_from_indices(patches_to_delete).to(device)

            # Apply deletion
            deleted_image = image.clone()
            for c in range(image.shape[0]):
                deleted_image[c] = deleted_image[c] * (1 - mask) + self.baseline_value * mask

            # Get prediction
            with torch.no_grad():
                pred = torch.softmax(model(deleted_image.unsqueeze(0)), dim=1)[0, true_label].item()

            scores.append(pred)
            fractions.append(n_patches_to_delete / n_patches)

        # Compute AUC using trapezoidal rule
        auc = np.trapz(scores, fractions)

        return {
            'auc': auc,
            'scores': scores,
            'fractions': fractions
        }


class ImageInsertionMetric:
    """
    Insertion metric for image explanations.

    Progressively adds features in order of importance (starting from baseline)
    and measures the area under the prediction score curve.
    """

    def __init__(self, patch_size: int = 56, image_size: int = 224,
                 baseline_value: float = 0.0, n_steps: int = 10):
        """
        Initialize Insertion metric.

        Args:
            patch_size: Size of each patch
            image_size: Size of the full image
            baseline_value: Value to use for baseline (non-inserted patches)
            n_steps: Number of insertion steps
        """
        self.patch_size = patch_size
        self.image_size = image_size
        self.baseline_value = baseline_value
        self.n_steps = n_steps

    def create_mask_from_indices(self, indices: List[int]) -> torch.Tensor:
        """Create a binary mask from patch indices."""
        n_patches_per_dim = self.image_size // self.patch_size
        mask = torch.zeros((self.image_size, self.image_size))

        for idx in indices:
            row = idx // n_patches_per_dim
            col = idx % n_patches_per_dim

            row_start = row * self.patch_size
            row_end = row_start + self.patch_size
            col_start = col * self.patch_size
            col_end = col_start + self.patch_size

            mask[row_start:row_end, col_start:col_end] = 1

        return mask

    def compute(self, model: nn.Module, image: torch.Tensor,
                attributions: torch.Tensor, true_label: int) -> Dict[str, Any]:
        """
        Compute insertion metric.

        Args:
            model: PyTorch model
            image: Input image (C, H, W)
            attributions: Feature importance scores (n_patches,)
            true_label: True class label

        Returns:
            Dictionary with 'auc' (area under curve), 'scores' (prediction scores
            at each step), and 'fractions' (fraction of features inserted)
        """
        device = image.device
        n_patches = len(attributions)

        # Sort patches by importance (descending)
        sorted_indices = torch.argsort(torch.abs(attributions), descending=True).cpu().numpy()

        # Start with all patches at baseline
        baseline_image = torch.ones_like(image) * self.baseline_value

        # Get baseline prediction
        with torch.no_grad():
            pred_baseline = torch.softmax(model(baseline_image.unsqueeze(0)), dim=1)[0, true_label].item()

        # Progressively insert patches
        scores = [pred_baseline]
        fractions = [0.0]

        patches_per_step = max(1, n_patches // self.n_steps)

        for step in range(1, self.n_steps + 1):
            # Determine which patches to insert
            n_patches_to_insert = min(step * patches_per_step, n_patches)
            patches_to_insert = sorted_indices[:n_patches_to_insert].tolist()

            # Create mask
            mask = self.create_mask_from_indices(patches_to_insert).to(device)

            # Apply insertion (keep original patches where mask=1, baseline elsewhere)
            inserted_image = baseline_image.clone()
            for c in range(image.shape[0]):
                inserted_image[c] = image[c] * mask + baseline_image[c] * (1 - mask)

            # Get prediction
            with torch.no_grad():
                pred = torch.softmax(model(inserted_image.unsqueeze(0)), dim=1)[0, true_label].item()

            scores.append(pred)
            fractions.append(n_patches_to_insert / n_patches)

        # Compute AUC using trapezoidal rule
        auc = np.trapz(scores, fractions)

        return {
            'auc': auc,
            'scores': scores,
            'fractions': fractions
        }


# =============================================================================
# Tabular Metrics
# =============================================================================

class TabularFaithfulnessPearson:
    """Faithfulness metric using Pearson correlation for tabular explanations."""

    def __init__(self, baseline_value: float = 0.0):
        """
        Initialize Faithfulness Pearson metric.

        Args:
            baseline_value: Value to use for ablated features
        """
        self.baseline_value = baseline_value

    def compute(self, model: Any, instance: np.ndarray,
                attributions: np.ndarray, true_label: Optional[int] = None) -> float:
        """
        Compute faithfulness using Pearson correlation.

        Args:
            model: Model (sklearn or PyTorch)
            instance: Input instance (n_features,)
            attributions: Feature importance scores (n_features,)
            true_label: True class label (if None, use predicted)

        Returns:
            Pearson correlation coefficient (higher is better)
        """
        n_features = len(attributions)

        # Get original prediction
        if hasattr(model, 'predict_proba'):
            pred_original = model.predict_proba(instance.reshape(1, -1))[0]
            if true_label is None:
                true_label = pred_original.argmax()
            pred_original = pred_original[true_label]
        else:
            with torch.no_grad():
                instance_tensor = torch.tensor(instance).float().unsqueeze(0)
                pred_original = torch.softmax(model(instance_tensor), dim=1)[0]
                if true_label is None:
                    true_label = pred_original.argmax().item()
                pred_original = pred_original[true_label].item()

        # For each feature, ablate it and measure prediction change
        prediction_changes = []
        attribution_scores = np.abs(attributions)

        for feature_idx in range(n_features):
            # Ablate the feature
            ablated_instance = instance.copy()
            ablated_instance[feature_idx] = self.baseline_value

            # Get prediction on ablated instance
            if hasattr(model, 'predict_proba'):
                pred_ablated = model.predict_proba(ablated_instance.reshape(1, -1))[0, true_label]
            else:
                with torch.no_grad():
                    ablated_tensor = torch.tensor(ablated_instance).float().unsqueeze(0)
                    pred_ablated = torch.softmax(model(ablated_tensor), dim=1)[0, true_label].item()

            # Record the change in prediction
            prediction_changes.append(abs(pred_original - pred_ablated))

        # Compute Pearson correlation
        if len(prediction_changes) > 1:
            corr, _ = pearsonr(attribution_scores, prediction_changes)
            return corr if not np.isnan(corr) else 0.0
        else:
            return 0.0


class TabularDeletionMetric:
    """Deletion metric for tabular explanations."""

    def __init__(self, baseline_value: float = 0.0, n_steps: int = 10):
        """
        Initialize Deletion metric.

        Args:
            baseline_value: Value to use for deleted features
            n_steps: Number of deletion steps
        """
        self.baseline_value = baseline_value
        self.n_steps = n_steps

    def compute(self, model: Any, instance: np.ndarray,
                attributions: np.ndarray, true_label: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute deletion metric.

        Args:
            model: Model (sklearn or PyTorch)
            instance: Input instance (n_features,)
            attributions: Feature importance scores (n_features,)
            true_label: True class label (if None, use predicted)

        Returns:
            Dictionary with 'auc', 'scores', and 'fractions'
        """
        n_features = len(attributions)

        # Sort features by importance (descending)
        sorted_indices = np.argsort(np.abs(attributions))[::-1]

        # Get original prediction
        if hasattr(model, 'predict_proba'):
            pred_original = model.predict_proba(instance.reshape(1, -1))[0]
            if true_label is None:
                true_label = pred_original.argmax()
            pred_original = pred_original[true_label]
        else:
            with torch.no_grad():
                instance_tensor = torch.tensor(instance).float().unsqueeze(0)
                pred_original = torch.softmax(model(instance_tensor), dim=1)[0]
                if true_label is None:
                    true_label = pred_original.argmax().item()
                pred_original = pred_original[true_label].item()

        # Progressively delete features
        scores = [pred_original]
        fractions = [0.0]

        features_per_step = max(1, n_features // self.n_steps)

        for step in range(1, self.n_steps + 1):
            # Determine which features to delete
            n_features_to_delete = min(step * features_per_step, n_features)
            features_to_delete = sorted_indices[:n_features_to_delete]

            # Apply deletion
            deleted_instance = instance.copy()
            deleted_instance[features_to_delete] = self.baseline_value

            # Get prediction
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(deleted_instance.reshape(1, -1))[0, true_label]
            else:
                with torch.no_grad():
                    deleted_tensor = torch.tensor(deleted_instance).float().unsqueeze(0)
                    pred = torch.softmax(model(deleted_tensor), dim=1)[0, true_label].item()

            scores.append(pred)
            fractions.append(n_features_to_delete / n_features)

        # Compute AUC
        auc = np.trapz(scores, fractions)

        return {
            'auc': auc,
            'scores': scores,
            'fractions': fractions
        }


class TabularInsertionMetric:
    """Insertion metric for tabular explanations."""

    def __init__(self, baseline_value: float = 0.0, n_steps: int = 10):
        """
        Initialize Insertion metric.

        Args:
            baseline_value: Value to use for baseline (non-inserted features)
            n_steps: Number of insertion steps
        """
        self.baseline_value = baseline_value
        self.n_steps = n_steps

    def compute(self, model: Any, instance: np.ndarray,
                attributions: np.ndarray, true_label: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute insertion metric.

        Args:
            model: Model (sklearn or PyTorch)
            instance: Input instance (n_features,)
            attributions: Feature importance scores (n_features,)
            true_label: True class label (if None, use predicted)

        Returns:
            Dictionary with 'auc', 'scores', and 'fractions'
        """
        n_features = len(attributions)

        # Sort features by importance (descending)
        sorted_indices = np.argsort(np.abs(attributions))[::-1]

        # Start with all features at baseline
        baseline_instance = np.full_like(instance, self.baseline_value)

        # Get baseline prediction
        if hasattr(model, 'predict_proba'):
            pred_baseline = model.predict_proba(baseline_instance.reshape(1, -1))[0]
            if true_label is None:
                # Use true label from original instance
                pred_orig = model.predict_proba(instance.reshape(1, -1))[0]
                true_label = pred_orig.argmax()
            pred_baseline = pred_baseline[true_label]
        else:
            with torch.no_grad():
                baseline_tensor = torch.tensor(baseline_instance).float().unsqueeze(0)
                pred_baseline = torch.softmax(model(baseline_tensor), dim=1)[0]
                if true_label is None:
                    instance_tensor = torch.tensor(instance).float().unsqueeze(0)
                    pred_orig = torch.softmax(model(instance_tensor), dim=1)[0]
                    true_label = pred_orig.argmax().item()
                pred_baseline = pred_baseline[true_label].item()

        # Progressively insert features
        scores = [pred_baseline]
        fractions = [0.0]

        features_per_step = max(1, n_features // self.n_steps)

        for step in range(1, self.n_steps + 1):
            # Determine which features to insert
            n_features_to_insert = min(step * features_per_step, n_features)
            features_to_insert = sorted_indices[:n_features_to_insert]

            # Apply insertion
            inserted_instance = baseline_instance.copy()
            inserted_instance[features_to_insert] = instance[features_to_insert]

            # Get prediction
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(inserted_instance.reshape(1, -1))[0, true_label]
            else:
                with torch.no_grad():
                    inserted_tensor = torch.tensor(inserted_instance).float().unsqueeze(0)
                    pred = torch.softmax(model(inserted_tensor), dim=1)[0, true_label].item()

            scores.append(pred)
            fractions.append(n_features_to_insert / n_features)

        # Compute AUC
        auc = np.trapz(scores, fractions)

        return {
            'auc': auc,
            'scores': scores,
            'fractions': fractions
        }


# =============================================================================
# Utility Functions
# =============================================================================

def compare_faithfulness_metrics(uncalibrated_results: Dict[str, float],
                                 calibrated_results: Dict[str, float]) -> Dict[str, float]:
    """
    Compare faithfulness metrics between uncalibrated and calibrated models.

    Args:
        uncalibrated_results: Dictionary of metric results for uncalibrated model
        calibrated_results: Dictionary of metric results for calibrated model

    Returns:
        Dictionary of improvement percentages
    """
    improvements = {}

    for metric_name in uncalibrated_results.keys():
        uncal_val = uncalibrated_results[metric_name]
        cal_val = calibrated_results[metric_name]

        # For Pearson correlation, higher is better
        if 'pearson' in metric_name.lower() or 'faithfulness' in metric_name.lower():
            if uncal_val != 0:
                improvement = ((cal_val - uncal_val) / abs(uncal_val)) * 100
            else:
                improvement = 0.0 if cal_val == 0 else 100.0

        # For Deletion AUC, lower is better (faster drop in prediction)
        elif 'deletion' in metric_name.lower():
            if uncal_val != 0:
                improvement = ((uncal_val - cal_val) / abs(uncal_val)) * 100
            else:
                improvement = 0.0 if cal_val == 0 else 100.0

        # For Insertion AUC, higher is better (faster rise in prediction)
        elif 'insertion' in metric_name.lower():
            if uncal_val != 0:
                improvement = ((cal_val - uncal_val) / abs(uncal_val)) * 100
            else:
                improvement = 0.0 if cal_val == 0 else 100.0

        else:
            improvement = 0.0

        improvements[metric_name] = improvement

    return improvements


if __name__ == "__main__":
    print("Faithfulness Metrics Module for MCal Experiments")
    print("="*60)
    print("Available metrics:")
    print("\nImage metrics:")
    print("  - ImageFaithfulnessPearson: Pearson correlation between attributions and predictions")
    print("  - ImageDeletionMetric: Progressive feature removal (AUC)")
    print("  - ImageInsertionMetric: Progressive feature addition (AUC)")
    print("\nTabular metrics:")
    print("  - TabularFaithfulnessPearson: Pearson correlation for tabular data")
    print("  - TabularDeletionMetric: Progressive feature removal (AUC)")
    print("  - TabularInsertionMetric: Progressive feature addition (AUC)")
    print("\nUsage example:")
    print("  from experiments.faithfulness_metrics import ImageFaithfulnessPearson")
    print("  faithfulness = ImageFaithfulnessPearson(patch_size=56)")
    print("  score = faithfulness.compute(model, image, attributions, true_label)")
