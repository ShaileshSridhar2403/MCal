#!/usr/bin/env python3
"""
Unified Feature Attribution Methods for MCal Experiments

A single file containing all explanation methods with a consistent API.
Includes LIME, SHAP, and other attribution techniques for vision, text, and tabular data.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    """Base class for all feature attribution explainers."""

    def __init__(self, model: nn.Module, **kwargs):
        """
        Initialize base explainer.

        Args:
            model: PyTorch model to explain
            **kwargs: Additional arguments specific to the explainer
        """
        self.model = model
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')

    @abstractmethod
    def explain_instance(self, input: torch.Tensor, label: Optional[int] = None) -> torch.Tensor:
        """
        Generate explanation for a single instance.

        Args:
            input: Input to explain
            label: Target class (if None, use predicted class)

        Returns:
            Feature importance scores
        """
        pass

    def explain_batch(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Generate explanations for a batch of inputs.

        Args:
            inputs: Batch of inputs
            labels: Target classes (if None, use predicted classes)

        Returns:
            List of feature importance scores
        """
        explanations = []
        for i in range(inputs.shape[0]):
            label = labels[i].item() if labels is not None else None
            explanation = self.explain_instance(inputs[i], label)
            explanations.append(explanation)
        return explanations


class ImageLIME(BaseExplainer):
    """LIME for image data using superpixel/patch segmentation."""

    def __init__(self, model: nn.Module, num_samples: int = 1000, patch_size: int = 56, image_size: int = 224):
        """
        Initialize Image LIME explainer.

        Args:
            model: PyTorch model that outputs probabilities
            num_samples: Number of perturbed samples for LIME
            patch_size: Size of each square patch
            image_size: Size of the input image
        """
        super().__init__(model)
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.image_size = image_size

    def create_square_patches(self) -> torch.Tensor:
        """
        Create grid of square patches.

        Returns:
            masks: Binary masks for each patch
        """
        n_patches_per_dim = self.image_size // self.patch_size
        n_patches = n_patches_per_dim * n_patches_per_dim

        masks = torch.zeros((n_patches, self.image_size, self.image_size))

        patch_idx = 0
        for i in range(n_patches_per_dim):
            for j in range(n_patches_per_dim):
                row_start = i * self.patch_size
                row_end = row_start + self.patch_size
                col_start = j * self.patch_size
                col_end = col_start + self.patch_size

                masks[patch_idx, row_start:row_end, col_start:col_end] = 1
                patch_idx += 1

        return masks

    def generate_perturbations(self, image: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate perturbed samples by randomly masking patches.

        Args:
            image: Input image (C, H, W)
            masks: Binary masks for patches

        Returns:
            perturbed_images: Perturbed samples
            binary_samples: Binary matrix indicating which patches are kept
        """
        n_patches = masks.shape[0]
        device = image.device

        # Generate random binary matrix for patch selection
        binary_samples = torch.randint(0, 2, size=(self.num_samples, n_patches))

        # Ensure at least one patch is kept in each sample
        for i in range(self.num_samples):
            if binary_samples[i].sum() == 0:
                binary_samples[i, torch.randint(0, n_patches, (1,)).item()] = 1

        # Generate perturbed images
        perturbed_images = []

        for i in range(self.num_samples):
            # Create combined mask for this sample
            combined_mask = torch.zeros((self.image_size, self.image_size), device=device)
            for j in range(n_patches):
                if binary_samples[i, j] == 1:  # Keep this patch
                    combined_mask += masks[j].to(device)

            # Apply mask to all channels (mask out by setting to 0/black)
            masked_image = image.clone()
            for c in range(image.shape[0]):  # Apply to all channels
                masked_image[c] = masked_image[c] * combined_mask

            perturbed_images.append(masked_image)

        perturbed_images = torch.stack(perturbed_images)

        return perturbed_images, binary_samples

    def explain_instance(self, image: torch.Tensor, label: Optional[int] = None) -> torch.Tensor:
        """
        Generate LIME explanation for a single image.

        Args:
            image: Input image (C, H, W)
            label: Class to explain (if None, use predicted class)

        Returns:
            importance_scores: Importance score for each patch
        """
        device = image.device

        # Create patches
        masks = self.create_square_patches()

        # Generate perturbations
        perturbed_images, binary_samples = self.generate_perturbations(image, masks)

        # Get predictions for perturbed samples
        with torch.no_grad():
            perturbed_images = perturbed_images.to(device)
            predictions = self.model(perturbed_images)

            # Get original prediction if label not specified
            if label is None:
                orig_pred = self.model(image.unsqueeze(0))
                label = orig_pred.argmax(dim=1).item()

            # Extract probabilities for target class
            probs = predictions[:, label].cpu()

        # Fit linear model using PyTorch
        # Weight samples by similarity (number of patches kept)
        weights = binary_samples.sum(dim=1).float() / binary_samples.shape[1]

        # Add bias term to binary samples
        X = torch.cat([binary_samples.float(), torch.ones(self.num_samples, 1)], dim=1)

        # Weighted least squares solution
        W = torch.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ probs

        # Add ridge regularization
        alpha = 1.0
        XtWX = XtWX + alpha * torch.eye(XtWX.shape[0])

        # Solve for coefficients
        coef = torch.linalg.solve(XtWX, XtWy)

        # Get feature importances (exclude bias term)
        importance_scores = coef[:-1]

        return importance_scores


class ImageKernelSHAP(BaseExplainer):
    """KernelSHAP for image data with patch-based masking."""

    def __init__(self, model: nn.Module, background_samples: Optional[torch.Tensor] = None,
                 num_samples: int = 100, patch_size: int = 56, image_size: int = 224):
        """
        Initialize Image KernelSHAP explainer.

        Args:
            model: PyTorch model that outputs probabilities
            background_samples: Background images for coalitions (if None, uses black image)
            num_samples: Number of samples for Shapley value estimation
            patch_size: Size of each square patch
            image_size: Size of the input image
        """
        super().__init__(model)
        self.background_samples = background_samples
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.image_size = image_size

    def create_square_patches(self) -> torch.Tensor:
        """
        Create grid of square patches.

        Returns:
            masks: Binary masks for each patch
        """
        n_patches_per_dim = self.image_size // self.patch_size
        n_patches = n_patches_per_dim * n_patches_per_dim

        masks = torch.zeros((n_patches, self.image_size, self.image_size))

        patch_idx = 0
        for i in range(n_patches_per_dim):
            for j in range(n_patches_per_dim):
                row_start = i * self.patch_size
                row_end = row_start + self.patch_size
                col_start = j * self.patch_size
                col_end = col_start + self.patch_size

                masks[patch_idx, row_start:row_end, col_start:col_end] = 1
                patch_idx += 1

        return masks

    def kernelshap_weight(self, num_features: int, num_present: int) -> float:
        """
        Compute KernelSHAP weight for a coalition.

        Args:
            num_features: Total number of features
            num_present: Number of features present in the coalition

        Returns:
            weight: KernelSHAP weight
        """
        if num_present == 0 or num_present == num_features:
            return 1e10  # Large weight for empty and full coalitions
        else:
            # Shapley kernel weight
            return (num_features - 1) / (num_present * (num_features - num_present))

    def explain_instance(self, image: torch.Tensor, label: Optional[int] = None) -> torch.Tensor:
        """
        Generate SHAP explanation for a single image using KernelSHAP approximation.

        Args:
            image: Input image (C, H, W)
            label: Class to explain (if None, use predicted class)

        Returns:
            shap_values: SHAP values for each patch
        """
        device = image.device

        # Create patches
        masks = self.create_square_patches()
        n_patches = masks.shape[0]

        # Get baseline (black image or mean of background samples)
        if self.background_samples is not None:
            baseline = self.background_samples.mean(dim=0)
        else:
            baseline = torch.zeros_like(image)

        # Get original prediction if label not specified
        with torch.no_grad():
            if label is None:
                orig_pred = self.model(image.unsqueeze(0))
                label = orig_pred.argmax(dim=1).item()

            # Get baseline prediction
            baseline_pred = self.model(baseline.unsqueeze(0))[0, label].item()

            # Get full image prediction
            full_pred = self.model(image.unsqueeze(0))[0, label].item()

        # Generate random coalitions
        coalitions = torch.randint(0, 2, size=(self.num_samples, n_patches))

        # Ensure we have empty and full coalitions
        coalitions[0, :] = 0  # Empty coalition
        coalitions[1, :] = 1  # Full coalition

        # Evaluate model on coalitions
        predictions = []
        weights = []

        for i in range(self.num_samples):
            # Create masked image based on coalition
            combined_mask = torch.zeros((self.image_size, self.image_size), device=device)
            num_present = 0

            for j in range(n_patches):
                if coalitions[i, j] == 1:
                    combined_mask += masks[j].to(device)
                    num_present += 1

            # Apply mask: use original where mask=1, baseline where mask=0
            masked_image = baseline.clone().to(device)
            for c in range(image.shape[0]):
                masked_image[c] = image[c] * combined_mask + baseline[c].to(device) * (1 - combined_mask)

            # Get prediction
            with torch.no_grad():
                pred = self.model(masked_image.unsqueeze(0))[0, label].item()
                predictions.append(pred)

            # Compute weight
            weight = self.kernelshap_weight(n_patches, num_present)
            weights.append(weight)

        # Convert to tensors
        predictions = torch.tensor(predictions)
        weights = torch.tensor(weights)
        coalitions = coalitions.float()

        # Normalize weights
        weights = weights / weights.sum()

        # Solve weighted least squares to get SHAP values
        # We want to find phi such that f(S) ≈ phi_0 + sum(phi_i * S_i)
        # Add bias term
        X = torch.cat([coalitions, torch.ones(self.num_samples, 1)], dim=1)

        # Weighted least squares
        W = torch.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ predictions

        # Add ridge regularization for stability
        alpha = 0.01
        XtWX = XtWX + alpha * torch.eye(XtWX.shape[0])

        # Solve for SHAP values
        phi = torch.linalg.solve(XtWX, XtWy)

        # Get SHAP values (exclude bias term)
        shap_values = phi[:-1]

        return shap_values


class TabularLIME(BaseExplainer):
    """LIME for tabular data."""

    def __init__(self, model: nn.Module, num_samples: int = 100, feature_names: Optional[List[str]] = None):
        """
        Initialize Tabular LIME explainer.

        Args:
            model: PyTorch model or sklearn-compatible model
            num_samples: Number of perturbed samples for LIME
            feature_names: Names of features (optional)
        """
        super().__init__(model)
        self.num_samples = num_samples
        self.feature_names = feature_names

    def generate_perturbations(self, instance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate perturbed samples by randomly perturbing features.

        Args:
            instance: Input instance (n_features,)

        Returns:
            perturbed_samples: Perturbed samples
            binary_samples: Binary matrix indicating which features are kept
        """
        n_features = instance.shape[0]

        # Generate random binary matrix for feature selection
        binary_samples = torch.randint(0, 2, size=(self.num_samples, n_features)).float()

        # Ensure at least one feature is kept in each sample
        for i in range(self.num_samples):
            if binary_samples[i].sum() == 0:
                binary_samples[i, torch.randint(0, n_features, (1,)).item()] = 1

        # Generate perturbed samples
        # For simplicity, we'll use mean imputation for missing features
        mean_values = torch.zeros_like(instance)  # Could be computed from training data

        perturbed_samples = []
        for i in range(self.num_samples):
            sample = instance.clone()
            # Replace features with mean where binary_samples[i] == 0
            mask = binary_samples[i] == 0
            sample[mask] = mean_values[mask]
            perturbed_samples.append(sample)

        perturbed_samples = torch.stack(perturbed_samples)

        return perturbed_samples, binary_samples

    def explain_instance(self, instance: torch.Tensor, label: Optional[int] = None) -> torch.Tensor:
        """
        Generate LIME explanation for a single tabular instance.

        Args:
            instance: Input instance (n_features,)
            label: Class to explain (if None, use predicted class)

        Returns:
            importance_scores: Importance score for each feature
        """
        # Generate perturbations
        perturbed_samples, binary_samples = self.generate_perturbations(instance)

        # Get predictions for perturbed samples
        with torch.no_grad():
            perturbed_samples = perturbed_samples.to(self.device)

            # Handle both PyTorch and sklearn models
            if hasattr(self.model, 'predict_proba'):
                # sklearn-style model
                predictions = torch.tensor(
                    self.model.predict_proba(perturbed_samples.cpu().numpy())
                ).float()
            else:
                # PyTorch model
                predictions = self.model(perturbed_samples)

            # Get original prediction if label not specified
            if label is None:
                if hasattr(self.model, 'predict_proba'):
                    orig_pred = torch.tensor(
                        self.model.predict_proba(instance.unsqueeze(0).cpu().numpy())
                    ).float()
                else:
                    orig_pred = self.model(instance.unsqueeze(0))
                label = orig_pred.argmax(dim=1).item()

            # Extract probabilities for target class
            if len(predictions.shape) > 1:
                probs = predictions[:, label].cpu()
            else:
                probs = predictions.cpu()

        # Fit linear model
        # Weight samples by similarity (Euclidean distance)
        distances = torch.norm(perturbed_samples - instance, dim=1)
        kernel_width = 0.25 * torch.sqrt(torch.tensor(instance.shape[0]).float())
        weights = torch.exp(-(distances ** 2) / (kernel_width ** 2))

        # Add bias term to binary samples
        X = torch.cat([binary_samples, torch.ones(self.num_samples, 1)], dim=1)

        # Weighted least squares solution
        W = torch.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ probs

        # Add ridge regularization
        alpha = 1.0
        XtWX = XtWX + alpha * torch.eye(XtWX.shape[0])

        # Solve for coefficients
        coef = torch.linalg.solve(XtWX, XtWy)

        # Get feature importances (exclude bias term)
        importance_scores = coef[:-1]

        return importance_scores


class TabularTreeSHAP:
    """TreeSHAP wrapper for tree-based models (XGBoost, RandomForest, etc.)."""

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize TreeSHAP explainer.

        Args:
            model: Tree-based model (XGBoost, RandomForest, etc.)
            feature_names: Names of features (optional)
        """
        self.model = model
        self.feature_names = feature_names

        # Try to import shap library
        try:
            import shap
            self.shap = shap
            self.explainer = shap.TreeExplainer(model)
        except ImportError:
            raise ImportError("Please install shap library: pip install shap")

    def explain_instance(self, instance: Union[np.ndarray, torch.Tensor], label: Optional[int] = None) -> np.ndarray:
        """
        Generate TreeSHAP explanation for a single instance.

        Args:
            instance: Input instance
            label: Not used for TreeSHAP (explains all outputs)

        Returns:
            shap_values: SHAP values for each feature
        """
        # Convert to numpy if needed
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()

        # Ensure correct shape
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)

        # Get SHAP values
        shap_values = self.explainer.shap_values(instance)

        # Handle multi-class output
        if isinstance(shap_values, list):
            if label is not None:
                shap_values = shap_values[label]
            else:
                # Use predicted class
                pred = self.model.predict(instance)
                if len(pred.shape) > 1:
                    label = pred.argmax()
                else:
                    label = int(pred[0])
                shap_values = shap_values[label]

        # Return first instance if batch
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        return shap_values


class TextLIME(BaseExplainer):
    """LIME for text data using token masking."""

    def __init__(self, model: nn.Module, tokenizer: Any, num_samples: int = 100):
        """
        Initialize Text LIME explainer.

        Args:
            model: PyTorch text model
            tokenizer: Tokenizer for the model
            num_samples: Number of perturbed samples for LIME
        """
        super().__init__(model)
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def generate_perturbations(self, tokens: List[str]) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Generate perturbed text samples by randomly masking tokens.

        Args:
            tokens: List of tokens

        Returns:
            perturbed_texts: List of perturbed token lists
            binary_samples: Binary matrix indicating which tokens are kept
        """
        n_tokens = len(tokens)

        # Generate random binary matrix for token selection
        binary_samples = torch.randint(0, 2, size=(self.num_samples, n_tokens))

        # Ensure at least one token is kept in each sample
        for i in range(self.num_samples):
            if binary_samples[i].sum() == 0:
                binary_samples[i, torch.randint(0, n_tokens, (1,)).item()] = 1

        # Generate perturbed texts
        perturbed_texts = []
        for i in range(self.num_samples):
            perturbed_tokens = []
            for j, token in enumerate(tokens):
                if binary_samples[i, j] == 1:
                    perturbed_tokens.append(token)
                # Skip masked tokens (or use [MASK] token if appropriate)

            perturbed_texts.append(perturbed_tokens)

        return perturbed_texts, binary_samples

    def explain_instance(self, text: str, label: Optional[int] = None) -> torch.Tensor:
        """
        Generate LIME explanation for text.

        Args:
            text: Input text string
            label: Class to explain (if None, use predicted class)

        Returns:
            importance_scores: Importance score for each token
        """
        # Tokenize text
        tokens = text.split()  # Simple whitespace tokenization

        # Generate perturbations
        perturbed_texts, binary_samples = self.generate_perturbations(tokens)

        # Get predictions for perturbed samples
        predictions = []
        for perturbed_tokens in perturbed_texts:
            perturbed_text = ' '.join(perturbed_tokens)

            # Encode and get prediction
            with torch.no_grad():
                encoded = self.tokenizer(perturbed_text, return_tensors='pt', padding=True, truncation=True)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.model(**encoded)

                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output

                probs = torch.softmax(logits, dim=-1)
                predictions.append(probs[0].cpu())

        predictions = torch.stack(predictions)

        # Get original prediction if label not specified
        if label is None:
            with torch.no_grad():
                encoded = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.model(**encoded)

                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output

                probs = torch.softmax(logits, dim=-1)
                label = probs.argmax(dim=-1).item()

        # Extract probabilities for target class
        probs = predictions[:, label]

        # Fit linear model
        # Weight samples by number of tokens kept
        weights = binary_samples.sum(dim=1).float() / binary_samples.shape[1]

        # Add bias term
        X = torch.cat([binary_samples.float(), torch.ones(self.num_samples, 1)], dim=1)

        # Weighted least squares
        W = torch.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ probs

        # Add ridge regularization
        alpha = 1.0
        XtWX = XtWX + alpha * torch.eye(XtWX.shape[0])

        # Solve for coefficients
        coef = torch.linalg.solve(XtWX, XtWy)

        # Get feature importances (exclude bias term)
        importance_scores = coef[:-1]

        return importance_scores




# Utility functions for patch/mask creation
def create_patch_mask_from_indices(indices: torch.Tensor, n_patches: int, patch_size: int = 56,
                                   image_size: int = 224) -> torch.Tensor:
    """
    Create a binary mask from patch indices.

    Args:
        indices: Indices of patches to keep
        n_patches: Total number of patches
        patch_size: Size of each patch
        image_size: Size of the full image

    Returns:
        Binary mask (H, W)
    """
    n_patches_per_dim = image_size // patch_size
    mask = torch.zeros((image_size, image_size))

    for idx in indices:
        idx = idx.item() if hasattr(idx, 'item') else idx
        row = idx // n_patches_per_dim
        col = idx % n_patches_per_dim

        row_start = row * patch_size
        row_end = row_start + patch_size
        col_start = col * patch_size
        col_end = col_start + patch_size

        mask[row_start:row_end, col_start:col_end] = 1

    return mask


def compute_distribution_shift(model: nn.Module, clean_images: torch.Tensor,
                               masked_images: torch.Tensor, n_classes: int = 4) -> float:
    """
    Compute KL divergence between prediction distributions.

    Args:
        model: Model to evaluate
        clean_images: Clean input images
        masked_images: Masked/ablated images
        n_classes: Number of classes

    Returns:
        KL divergence value
    """
    clean_preds = []
    masked_preds = []

    with torch.no_grad():
        for clean, masked in zip(clean_images, masked_images):
            clean_pred = model(clean.unsqueeze(0))
            masked_pred = model(masked.unsqueeze(0))

            clean_preds.append(clean_pred.argmax().item())
            masked_preds.append(masked_pred.argmax().item())

    # Compute class distributions
    clean_dist = torch.bincount(torch.tensor(clean_preds), minlength=n_classes).float()
    masked_dist = torch.bincount(torch.tensor(masked_preds), minlength=n_classes).float()

    # Normalize
    clean_dist = clean_dist / clean_dist.sum()
    masked_dist = masked_dist / masked_dist.sum()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    masked_dist = masked_dist + eps
    clean_dist = clean_dist + eps

    # KL divergence
    kl = torch.nn.functional.kl_div(masked_dist.log(), clean_dist, reduction='sum')
    return kl.item()


if __name__ == "__main__":
    print("Unified Feature Attribution Methods Module")
    print("="*50)
    print("Available explainers:")
    print("  - ImageLIME: LIME for images using patch segmentation")
    print("  - ImageKernelSHAP: KernelSHAP for images")
    print("  - TabularLIME: LIME for tabular data")
    print("  - TabularTreeSHAP: TreeSHAP for tree-based models")
    print("  - TextLIME: LIME for text using token masking")
    print("\nUsage example:")
    print("  from experiments.explanations import ImageLIME, ImageKernelSHAP")
    print("  explainer = ImageLIME(model, num_samples=100)")
    print("  attribution = explainer.explain_instance(image)")