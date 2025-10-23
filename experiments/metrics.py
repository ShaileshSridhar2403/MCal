#!/usr/bin/env python3
"""
Evaluation Metrics for MCal Experiments

Contains metrics for evaluating explanation quality and model calibration:
- KL Divergence (Missingness Bias)
- Sufficiency
- Comprehensiveness

Implementations for different modalities: image, text, and tabular data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        """Compute the metric value."""
        pass


# =============================================================================
# Image Metrics
# =============================================================================

class ImageKLDivergence:
    """KL Divergence metric for measuring missingness bias in image models."""

    def __init__(self, n_classes: int = 4):
        """
        Initialize KL Divergence metric.

        Args:
            n_classes: Number of classes in the dataset
        """
        self.n_classes = n_classes

    def compute(self, model: nn.Module, clean_images: torch.Tensor,
                ablated_images: torch.Tensor) -> float:
        """
        Compute KL divergence between class distributions on clean vs ablated images.

        Args:
            model: PyTorch model
            clean_images: Clean/original images (N, C, H, W)
            ablated_images: Ablated/masked images (N, C, H, W)

        Returns:
            KL divergence value (lower is better)
        """
        clean_preds = []
        ablated_preds = []

        with torch.no_grad():
            # Process in batches if needed
            batch_size = 32
            for i in range(0, len(clean_images), batch_size):
                clean_batch = clean_images[i:i+batch_size]
                ablated_batch = ablated_images[i:i+batch_size]

                # Get predictions
                clean_logits = model(clean_batch)
                ablated_logits = model(ablated_batch)

                clean_preds.extend(clean_logits.argmax(dim=1).cpu().tolist())
                ablated_preds.extend(ablated_logits.argmax(dim=1).cpu().tolist())

        # Compute class distributions
        clean_dist = torch.bincount(torch.tensor(clean_preds), minlength=self.n_classes).float()
        ablated_dist = torch.bincount(torch.tensor(ablated_preds), minlength=self.n_classes).float()

        # Normalize
        clean_dist = clean_dist / clean_dist.sum()
        ablated_dist = ablated_dist / ablated_dist.sum()

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        ablated_dist = ablated_dist + eps
        clean_dist = clean_dist + eps

        # Compute KL divergence: KL(ablated || clean)
        kl_div = F.kl_div(ablated_dist.log(), clean_dist, reduction='sum')

        return kl_div.item()


class ImageSufficiency:
    """Sufficiency metric for image explanations."""

    def __init__(self, patch_size: int = 56, image_size: int = 224):
        """
        Initialize Sufficiency metric.

        Args:
            patch_size: Size of each patch
            image_size: Size of the full image
        """
        self.patch_size = patch_size
        self.image_size = image_size

    def create_mask_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Create a binary mask from patch indices.

        Args:
            indices: Indices of patches to keep

        Returns:
            Binary mask (H, W)
        """
        n_patches_per_dim = self.image_size // self.patch_size
        mask = torch.zeros((self.image_size, self.image_size))

        for idx in indices:
            idx = idx.item() if hasattr(idx, 'item') else idx
            row = idx // n_patches_per_dim
            col = idx % n_patches_per_dim

            row_start = row * self.patch_size
            row_end = row_start + self.patch_size
            col_start = col * self.patch_size
            col_end = col_start + self.patch_size

            mask[row_start:row_end, col_start:col_end] = 1

        return mask

    def compute(self, model: nn.Module, image: torch.Tensor,
                attributions: torch.Tensor, top_k: int,
                true_label: int) -> float:
        """
        Compute sufficiency: confidence drop when keeping only top-k important features.

        Args:
            model: PyTorch model
            image: Input image (C, H, W)
            attributions: Feature importance scores (n_patches,)
            top_k: Number of top features to keep
            true_label: True class label

        Returns:
            Sufficiency score (lower is better)
        """
        device = image.device

        # Get top-k most important patches
        top_k_indices = torch.argsort(torch.abs(attributions), descending=True)[:top_k]

        # Create mask keeping only top-k patches
        mask = self.create_mask_from_indices(top_k_indices).to(device)

        # Apply mask to image (set non-important regions to black)
        masked_image = image.clone()
        for c in range(image.shape[0]):
            masked_image[c] = masked_image[c] * mask

        # Compare predictions
        with torch.no_grad():
            pred_original = model(image.unsqueeze(0))[0, true_label]
            pred_masked = model(masked_image.unsqueeze(0))[0, true_label]

        # Sufficiency = drop in confidence
        sufficiency = (pred_original - pred_masked).item()

        return sufficiency


class ImageComprehensiveness:
    """Comprehensiveness metric for image explanations."""

    def __init__(self, patch_size: int = 56, image_size: int = 224):
        """
        Initialize Comprehensiveness metric.

        Args:
            patch_size: Size of each patch
            image_size: Size of the full image
        """
        self.patch_size = patch_size
        self.image_size = image_size

    def create_mask_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Create a binary mask from patch indices.

        Args:
            indices: Indices of patches to remove

        Returns:
            Binary mask (H, W)
        """
        n_patches_per_dim = self.image_size // self.patch_size
        mask = torch.ones((self.image_size, self.image_size))

        for idx in indices:
            idx = idx.item() if hasattr(idx, 'item') else idx
            row = idx // n_patches_per_dim
            col = idx % n_patches_per_dim

            row_start = row * self.patch_size
            row_end = row_start + self.patch_size
            col_start = col * self.patch_size
            col_end = col_start + self.patch_size

            mask[row_start:row_end, col_start:col_end] = 0

        return mask

    def compute(self, model: nn.Module, image: torch.Tensor,
                attributions: torch.Tensor, top_k: int,
                true_label: int) -> float:
        """
        Compute comprehensiveness: confidence drop when removing top-k important features.

        Args:
            model: PyTorch model
            image: Input image (C, H, W)
            attributions: Feature importance scores (n_patches,)
            top_k: Number of top features to remove
            true_label: True class label

        Returns:
            Comprehensiveness score (higher is better)
        """
        device = image.device

        # Get top-k most important patches
        top_k_indices = torch.argsort(torch.abs(attributions), descending=True)[:top_k]

        # Create mask removing top-k patches
        mask = self.create_mask_from_indices(top_k_indices).to(device)

        # Apply mask to image (set important regions to black)
        masked_image = image.clone()
        for c in range(image.shape[0]):
            masked_image[c] = masked_image[c] * mask

        # Compare predictions
        with torch.no_grad():
            pred_original = model(image.unsqueeze(0))[0, true_label]
            pred_masked = model(masked_image.unsqueeze(0))[0, true_label]

        # Comprehensiveness = drop in confidence
        comprehensiveness = (pred_original - pred_masked).item()

        return comprehensiveness


# =============================================================================
# Tabular Metrics
# =============================================================================

class TabularKLDivergence:
    """KL Divergence metric for measuring missingness bias in tabular models."""

    def __init__(self, n_classes: int = 2):
        """
        Initialize KL Divergence metric.

        Args:
            n_classes: Number of classes in the dataset
        """
        self.n_classes = n_classes

    def compute(self, model: Any, clean_data: np.ndarray,
                ablated_data: np.ndarray) -> float:
        """
        Compute KL divergence between class distributions on clean vs ablated data.

        Args:
            model: Model (sklearn or PyTorch)
            clean_data: Clean/original data (N, n_features)
            ablated_data: Ablated/masked data (N, n_features)

        Returns:
            KL divergence value (lower is better)
        """
        # Handle both PyTorch and sklearn models
        if hasattr(model, 'predict'):
            # sklearn-style model
            clean_preds = model.predict(clean_data)
            ablated_preds = model.predict(ablated_data)
        else:
            # PyTorch model
            with torch.no_grad():
                clean_tensor = torch.tensor(clean_data).float()
                ablated_tensor = torch.tensor(ablated_data).float()

                clean_logits = model(clean_tensor)
                ablated_logits = model(ablated_tensor)

                clean_preds = clean_logits.argmax(dim=1).cpu().numpy()
                ablated_preds = ablated_logits.argmax(dim=1).cpu().numpy()

        # Compute class distributions
        clean_dist = np.bincount(clean_preds, minlength=self.n_classes).astype(float)
        ablated_dist = np.bincount(ablated_preds, minlength=self.n_classes).astype(float)

        # Normalize
        clean_dist = clean_dist / clean_dist.sum()
        ablated_dist = ablated_dist / ablated_dist.sum()

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        ablated_dist = ablated_dist + eps
        clean_dist = clean_dist + eps

        # Compute KL divergence
        kl_div = np.sum(ablated_dist * np.log(ablated_dist / clean_dist))

        return kl_div


class TabularSufficiency:
    """Sufficiency metric for tabular explanations."""

    def __init__(self, baseline_value: float = 0.0):
        """
        Initialize Sufficiency metric.

        Args:
            baseline_value: Value to use for masked features
        """
        self.baseline_value = baseline_value

    def compute(self, model: Any, instance: np.ndarray,
                attributions: np.ndarray, top_k: int,
                true_label: Optional[int] = None) -> float:
        """
        Compute sufficiency: confidence drop when keeping only top-k important features.

        Args:
            model: Model (sklearn or PyTorch)
            instance: Input instance (n_features,)
            attributions: Feature importance scores (n_features,)
            top_k: Number of top features to keep
            true_label: True class label (if None, use predicted)

        Returns:
            Sufficiency score (lower is better)
        """
        # Get top-k most important features
        top_k_indices = np.argsort(np.abs(attributions))[-top_k:]

        # Create masked instance (keep only top-k features)
        masked_instance = np.full_like(instance, self.baseline_value)
        masked_instance[top_k_indices] = instance[top_k_indices]

        # Get predictions
        if hasattr(model, 'predict_proba'):
            # sklearn-style model
            pred_original = model.predict_proba(instance.reshape(1, -1))[0]
            pred_masked = model.predict_proba(masked_instance.reshape(1, -1))[0]

            if true_label is None:
                true_label = pred_original.argmax()

            sufficiency = pred_original[true_label] - pred_masked[true_label]
        else:
            # PyTorch model
            with torch.no_grad():
                instance_tensor = torch.tensor(instance).float().unsqueeze(0)
                masked_tensor = torch.tensor(masked_instance).float().unsqueeze(0)

                pred_original = torch.softmax(model(instance_tensor), dim=1)[0]
                pred_masked = torch.softmax(model(masked_tensor), dim=1)[0]

                if true_label is None:
                    true_label = pred_original.argmax().item()

                sufficiency = (pred_original[true_label] - pred_masked[true_label]).item()

        return sufficiency


class TabularComprehensiveness:
    """Comprehensiveness metric for tabular explanations."""

    def __init__(self, baseline_value: float = 0.0):
        """
        Initialize Comprehensiveness metric.

        Args:
            baseline_value: Value to use for masked features
        """
        self.baseline_value = baseline_value

    def compute(self, model: Any, instance: np.ndarray,
                attributions: np.ndarray, top_k: int,
                true_label: Optional[int] = None) -> float:
        """
        Compute comprehensiveness: confidence drop when removing top-k important features.

        Args:
            model: Model (sklearn or PyTorch)
            instance: Input instance (n_features,)
            attributions: Feature importance scores (n_features,)
            top_k: Number of top features to remove
            true_label: True class label (if None, use predicted)

        Returns:
            Comprehensiveness score (higher is better)
        """
        # Get top-k most important features
        top_k_indices = np.argsort(np.abs(attributions))[-top_k:]

        # Create masked instance (remove top-k features)
        masked_instance = instance.copy()
        masked_instance[top_k_indices] = self.baseline_value

        # Get predictions
        if hasattr(model, 'predict_proba'):
            # sklearn-style model
            pred_original = model.predict_proba(instance.reshape(1, -1))[0]
            pred_masked = model.predict_proba(masked_instance.reshape(1, -1))[0]

            if true_label is None:
                true_label = pred_original.argmax()

            comprehensiveness = pred_original[true_label] - pred_masked[true_label]
        else:
            # PyTorch model
            with torch.no_grad():
                instance_tensor = torch.tensor(instance).float().unsqueeze(0)
                masked_tensor = torch.tensor(masked_instance).float().unsqueeze(0)

                pred_original = torch.softmax(model(instance_tensor), dim=1)[0]
                pred_masked = torch.softmax(model(masked_tensor), dim=1)[0]

                if true_label is None:
                    true_label = pred_original.argmax().item()

                comprehensiveness = (pred_original[true_label] - pred_masked[true_label]).item()

        return comprehensiveness


# =============================================================================
# Text Metrics
# =============================================================================

class TextKLDivergence:
    """KL Divergence metric for measuring missingness bias in text models."""

    def __init__(self, n_classes: int = 2):
        """
        Initialize KL Divergence metric.

        Args:
            n_classes: Number of classes in the dataset
        """
        self.n_classes = n_classes

    def compute(self, model: nn.Module, tokenizer: Any,
                clean_texts: List[str], ablated_texts: List[str]) -> float:
        """
        Compute KL divergence between class distributions on clean vs ablated texts.

        Args:
            model: PyTorch text model
            tokenizer: Tokenizer for the model
            clean_texts: Clean/original texts
            ablated_texts: Ablated/masked texts

        Returns:
            KL divergence value (lower is better)
        """
        clean_preds = []
        ablated_preds = []

        with torch.no_grad():
            for clean_text, ablated_text in zip(clean_texts, ablated_texts):
                # Tokenize
                clean_encoded = tokenizer(clean_text, return_tensors='pt',
                                         padding=True, truncation=True)
                ablated_encoded = tokenizer(ablated_text, return_tensors='pt',
                                           padding=True, truncation=True)

                # Move to device
                device = next(model.parameters()).device
                clean_encoded = {k: v.to(device) for k, v in clean_encoded.items()}
                ablated_encoded = {k: v.to(device) for k, v in ablated_encoded.items()}

                # Get predictions
                clean_output = model(**clean_encoded)
                ablated_output = model(**ablated_encoded)

                if hasattr(clean_output, 'logits'):
                    clean_logits = clean_output.logits
                    ablated_logits = ablated_output.logits
                else:
                    clean_logits = clean_output
                    ablated_logits = ablated_output

                clean_preds.append(clean_logits.argmax(dim=1).cpu().item())
                ablated_preds.append(ablated_logits.argmax(dim=1).cpu().item())

        # Compute class distributions
        clean_dist = torch.bincount(torch.tensor(clean_preds), minlength=self.n_classes).float()
        ablated_dist = torch.bincount(torch.tensor(ablated_preds), minlength=self.n_classes).float()

        # Normalize
        clean_dist = clean_dist / clean_dist.sum()
        ablated_dist = ablated_dist / ablated_dist.sum()

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        ablated_dist = ablated_dist + eps
        clean_dist = clean_dist + eps

        # Compute KL divergence
        kl_div = F.kl_div(ablated_dist.log(), clean_dist, reduction='sum')

        return kl_div.item()


class TextSufficiency:
    """Sufficiency metric for text explanations."""

    def __init__(self, mask_token: str = '[MASK]'):
        """
        Initialize Sufficiency metric.

        Args:
            mask_token: Token to use for masking
        """
        self.mask_token = mask_token

    def compute(self, model: nn.Module, tokenizer: Any,
                text: str, tokens: List[str],
                attributions: np.ndarray, top_k: int,
                true_label: Optional[int] = None) -> float:
        """
        Compute sufficiency: confidence drop when keeping only top-k important tokens.

        Args:
            model: PyTorch text model
            tokenizer: Tokenizer for the model
            text: Original text
            tokens: List of tokens
            attributions: Token importance scores
            top_k: Number of top tokens to keep
            true_label: True class label (if None, use predicted)

        Returns:
            Sufficiency score (lower is better)
        """
        # Get top-k most important tokens
        top_k_indices = np.argsort(np.abs(attributions))[-top_k:]

        # Create masked text (keep only top-k tokens)
        masked_tokens = []
        for i, token in enumerate(tokens):
            if i in top_k_indices:
                masked_tokens.append(token)
            # Skip other tokens or replace with mask

        masked_text = ' '.join(masked_tokens)

        # Get predictions
        with torch.no_grad():
            # Original text
            encoded_orig = tokenizer(text, return_tensors='pt',
                                    padding=True, truncation=True)
            device = next(model.parameters()).device
            encoded_orig = {k: v.to(device) for k, v in encoded_orig.items()}
            output_orig = model(**encoded_orig)

            # Masked text
            encoded_mask = tokenizer(masked_text, return_tensors='pt',
                                    padding=True, truncation=True)
            encoded_mask = {k: v.to(device) for k, v in encoded_mask.items()}
            output_mask = model(**encoded_mask)

            if hasattr(output_orig, 'logits'):
                logits_orig = output_orig.logits
                logits_mask = output_mask.logits
            else:
                logits_orig = output_orig
                logits_mask = output_mask

            probs_orig = torch.softmax(logits_orig, dim=-1)[0]
            probs_mask = torch.softmax(logits_mask, dim=-1)[0]

            if true_label is None:
                true_label = probs_orig.argmax().item()

            sufficiency = (probs_orig[true_label] - probs_mask[true_label]).item()

        return sufficiency


class TextComprehensiveness:
    """Comprehensiveness metric for text explanations."""

    def __init__(self, mask_token: str = '[MASK]'):
        """
        Initialize Comprehensiveness metric.

        Args:
            mask_token: Token to use for masking
        """
        self.mask_token = mask_token

    def compute(self, model: nn.Module, tokenizer: Any,
                text: str, tokens: List[str],
                attributions: np.ndarray, top_k: int,
                true_label: Optional[int] = None) -> float:
        """
        Compute comprehensiveness: confidence drop when removing top-k important tokens.

        Args:
            model: PyTorch text model
            tokenizer: Tokenizer for the model
            text: Original text
            tokens: List of tokens
            attributions: Token importance scores
            top_k: Number of top tokens to remove
            true_label: True class label (if None, use predicted)

        Returns:
            Comprehensiveness score (higher is better)
        """
        # Get top-k most important tokens
        top_k_indices = np.argsort(np.abs(attributions))[-top_k:]

        # Create masked text (remove top-k tokens)
        masked_tokens = []
        for i, token in enumerate(tokens):
            if i not in top_k_indices:
                masked_tokens.append(token)
            # Skip important tokens

        masked_text = ' '.join(masked_tokens) if masked_tokens else self.mask_token

        # Get predictions
        with torch.no_grad():
            # Original text
            encoded_orig = tokenizer(text, return_tensors='pt',
                                    padding=True, truncation=True)
            device = next(model.parameters()).device
            encoded_orig = {k: v.to(device) for k, v in encoded_orig.items()}
            output_orig = model(**encoded_orig)

            # Masked text
            encoded_mask = tokenizer(masked_text, return_tensors='pt',
                                    padding=True, truncation=True)
            encoded_mask = {k: v.to(device) for k, v in encoded_mask.items()}
            output_mask = model(**encoded_mask)

            if hasattr(output_orig, 'logits'):
                logits_orig = output_orig.logits
                logits_mask = output_mask.logits
            else:
                logits_orig = output_orig
                logits_mask = output_mask

            probs_orig = torch.softmax(logits_orig, dim=-1)[0]
            probs_mask = torch.softmax(logits_mask, dim=-1)[0]

            if true_label is None:
                true_label = probs_orig.argmax().item()

            comprehensiveness = (probs_orig[true_label] - probs_mask[true_label]).item()

        return comprehensiveness


# =============================================================================
# Utility Functions
# =============================================================================


def compute_metric_improvement(uncalibrated_value: float, calibrated_value: float,
                               metric_type: str) -> float:
    """
    Compute percentage improvement from calibration.

    Args:
        uncalibrated_value: Metric value without calibration
        calibrated_value: Metric value with calibration
        metric_type: Type of metric ('kl_divergence', 'sufficiency', 'comprehensiveness')

    Returns:
        Percentage improvement (positive means calibration helped)
    """
    if metric_type in ['kl_divergence', 'sufficiency']:
        # Lower is better for these metrics
        if uncalibrated_value == 0:
            return 0.0
        improvement = (1 - calibrated_value / uncalibrated_value) * 100
    elif metric_type == 'comprehensiveness':
        # Higher is better for comprehensiveness
        if uncalibrated_value == 0:
            return 0.0 if calibrated_value == 0 else 100.0
        improvement = (calibrated_value / uncalibrated_value - 1) * 100
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return improvement


if __name__ == "__main__":
    print("Evaluation Metrics Module for MCal Experiments")
    print("="*50)
    print("Available metrics:")
    print("\nImage metrics:")
    print("  - ImageKLDivergence: Measure missingness bias")
    print("  - ImageSufficiency: Drop when keeping top-k features")
    print("  - ImageComprehensiveness: Drop when removing top-k features")
    print("\nTabular metrics:")
    print("  - TabularKLDivergence: Measure missingness bias")
    print("  - TabularSufficiency: Drop when keeping top-k features")
    print("  - TabularComprehensiveness: Drop when removing top-k features")
    print("\nText metrics:")
    print("  - TextKLDivergence: Measure missingness bias")
    print("  - TextSufficiency: Drop when keeping top-k tokens")
    print("  - TextComprehensiveness: Drop when removing top-k tokens")
    print("\nUsage example:")
    print("  from experiments.metrics import ImageKLDivergence, ImageSufficiency")
    print("  kl_metric = ImageKLDivergence(n_classes=4)")
    print("  kl_value = kl_metric.compute(model, clean_images, ablated_images)")