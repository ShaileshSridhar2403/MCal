"""Evaluation metrics for calibration assessment."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def compute_kl_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray,
    fractions: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute KL divergence metrics for calibration evaluation.
    
    This function replicates the KL divergence computation from the original
    get_benchmarks.py but in a modular, reusable way.
    
    Args:
        predictions: Predicted probability distributions of shape (n_fractions, n_samples, n_classes)
                    or (n_samples, n_classes)
        targets: Target probability distributions of same shape as predictions
        fractions: Array of fraction values (default: linspace(0, 1, n_fractions))
        
    Returns:
        Dictionary containing KL divergence metrics
    """
    # Handle 2D input (single fraction)
    if predictions.ndim == 2:
        predictions = predictions[np.newaxis, ...]
        targets = targets[np.newaxis, ...]
    
    n_fractions, n_samples, n_classes = predictions.shape
    
    if fractions is None:
        fractions = np.linspace(0, 1, n_fractions)
    
    # Initialize storage for fraction-wise results
    fractionwise_kl_prob = []
    fractionwise_kl_argmax = []
    
    for i in range(n_fractions):
        pred_fraction = predictions[i]
        target_fraction = targets[i]
        
        # Compute probability-based expectation
        prob_expectation_pred = pred_fraction.mean(axis=0)
        prob_expectation_target = target_fraction.mean(axis=0)
        
        # Normalize to ensure valid probability distributions
        prob_expectation_pred = prob_expectation_pred / prob_expectation_pred.sum()
        prob_expectation_target = prob_expectation_target / prob_expectation_target.sum()
        
        # Compute argmax-based expectation
        argmax_pred = np.eye(n_classes)[pred_fraction.argmax(axis=1)]
        argmax_target = np.eye(n_classes)[target_fraction.argmax(axis=1)]
        
        argmax_expectation_pred = argmax_pred.mean(axis=0)
        argmax_expectation_target = argmax_target.mean(axis=0)
        
        # Handle zero values in argmax expectations
        epsilon = 1e-9
        argmax_expectation_pred = np.maximum(argmax_expectation_pred, epsilon)
        argmax_expectation_target = np.maximum(argmax_expectation_target, epsilon)
        
        # Normalize argmax expectations
        argmax_expectation_pred = argmax_expectation_pred / argmax_expectation_pred.sum()
        argmax_expectation_target = argmax_expectation_target / argmax_expectation_target.sum()
        
        # Compute KL divergences
        kl_prob = kl_divergence(prob_expectation_target, prob_expectation_pred)
        kl_argmax = kl_divergence(argmax_expectation_target, argmax_expectation_pred)
        
        fractionwise_kl_prob.append(kl_prob)
        fractionwise_kl_argmax.append(kl_argmax)
    
    # Compute averages
    average_kl_prob = np.mean(fractionwise_kl_prob)
    average_kl_argmax = np.mean(fractionwise_kl_argmax)
    
    return {
        'average_kl_prob': average_kl_prob,
        'average_kl_argmax': average_kl_argmax,
        'fractionwise_kl': list(zip(fractionwise_kl_argmax, fractionwise_kl_prob)),
        'fractionwise_kl_prob': fractionwise_kl_prob,
        'fractionwise_kl_argmax': fractionwise_kl_argmax,
        'fractions': fractions.tolist() if fractions is not None else None
    }


def kl_divergence(P: np.ndarray, Q: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute KL divergence D(P||Q) between two probability distributions.
    
    Args:
        P: First probability distribution
        Q: Second probability distribution  
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence value
    """
    # Add epsilon to avoid log(0)
    P = np.maximum(P, epsilon)
    Q = np.maximum(Q, epsilon)
    
    # Normalize to ensure they sum to 1
    P = P / P.sum()
    Q = Q / Q.sum()
    
    return np.sum(P * np.log(P / Q))


def compute_calibration_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute calibration metrics including ECE, MCE, and reliability diagrams.
    
    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        targets: Target probabilities of same shape
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary containing calibration metrics
    """
    if predictions.ndim == 3:
        # If 3D, use the first fraction or average across fractions
        predictions = predictions[0] if predictions.shape[0] == 1 else predictions.mean(axis=0)
        targets = targets[0] if targets.shape[0] == 1 else targets.mean(axis=0)
    
    n_samples, n_classes = predictions.shape
    
    # Convert to class labels for some metrics
    pred_labels = predictions.argmax(axis=1)
    true_labels = targets.argmax(axis=1)
    
    # Get max probabilities (confidence scores)
    confidences = predictions.max(axis=1)
    correct = (pred_labels == true_labels).astype(int)
    
    # Compute Expected Calibration Error (ECE)
    ece = expected_calibration_error(confidences, correct, n_bins)
    
    # Compute Maximum Calibration Error (MCE)
    mce = maximum_calibration_error(confidences, correct, n_bins)
    
    # Compute Brier Score (if we have probabilistic targets)
    try:
        # For multi-class, compute average Brier score
        brier_scores = []
        for class_idx in range(n_classes):
            true_binary = (targets.argmax(axis=1) == class_idx).astype(int)
            pred_binary = predictions[:, class_idx]
            brier_scores.append(brier_score_loss(true_binary, pred_binary))
        brier_score = np.mean(brier_scores)
    except:
        brier_score = None
    
    # Compute log loss
    try:
        # Ensure predictions are valid probabilities
        pred_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        pred_normalized = pred_clipped / pred_clipped.sum(axis=1, keepdims=True)
        log_loss_score = log_loss(true_labels, pred_normalized)
    except:
        log_loss_score = None
    
    # Compute reliability diagram data
    reliability_data = compute_reliability_diagram(confidences, correct, n_bins)
    
    return {
        'ece': ece,
        'mce': mce,
        'brier_score': brier_score,
        'log_loss': log_loss_score,
        'reliability_diagram': reliability_data,
        'n_bins': n_bins
    }


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: Confidence scores for predictions
        correct: Binary array indicating correct predictions
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(confidences)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum() / total_samples
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Maximum Calibration Error (MCE).
    
    Args:
        confidences: Confidence scores for predictions
        correct: Binary array indicating correct predictions
        n_bins: Number of bins
        
    Returns:
        MCE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_error = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_error = max(max_error, error)
    
    return max_error


def compute_reliability_diagram(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10
) -> Dict[str, List[float]]:
    """Compute data for reliability diagram.
    
    Args:
        confidences: Confidence scores for predictions
        correct: Binary array indicating correct predictions
        n_bins: Number of bins
        
    Returns:
        Dictionary with bin centers, accuracies, confidences, and counts
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    accuracies = []
    avg_confidences = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count_in_bin = in_bin.sum()
        
        if count_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
        else:
            accuracy_in_bin = 0
            avg_confidence_in_bin = (bin_lower + bin_upper) / 2
        
        accuracies.append(accuracy_in_bin)
        avg_confidences.append(avg_confidence_in_bin)
        counts.append(count_in_bin)
    
    return {
        'bin_centers': bin_centers.tolist(),
        'accuracies': accuracies,
        'confidences': avg_confidences,
        'counts': counts
    }


def compute_distributional_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, Any]:
    """Compute distributional distance metrics.
    
    Args:
        predictions: Predicted probability distributions
        targets: Target probability distributions
        
    Returns:
        Dictionary containing distributional metrics
    """
    if predictions.ndim == 3:
        # Average across fractions if 3D
        predictions = predictions.mean(axis=0)
        targets = targets.mean(axis=0)
    
    # Compute various distributional distances
    metrics = {}
    
    # L1 (Total Variation) distance
    l1_distances = np.abs(predictions - targets).sum(axis=1)
    metrics['l1_distance'] = {
        'mean': l1_distances.mean(),
        'std': l1_distances.std(),
        'median': np.median(l1_distances)
    }
    
    # L2 (Euclidean) distance
    l2_distances = np.linalg.norm(predictions - targets, axis=1)
    metrics['l2_distance'] = {
        'mean': l2_distances.mean(),
        'std': l2_distances.std(),
        'median': np.median(l2_distances)
    }
    
    # Hellinger distance
    hellinger_distances = []
    for i in range(len(predictions)):
        # Ensure non-negative values
        p = np.maximum(predictions[i], 1e-15)
        q = np.maximum(targets[i], 1e-15)
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        # Compute Hellinger distance
        h_dist = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))
        hellinger_distances.append(h_dist)
    
    hellinger_distances = np.array(hellinger_distances)
    metrics['hellinger_distance'] = {
        'mean': hellinger_distances.mean(),
        'std': hellinger_distances.std(),
        'median': np.median(hellinger_distances)
    }
    
    # Jensen-Shannon divergence
    js_divergences = []
    for i in range(len(predictions)):
        p = np.maximum(predictions[i], 1e-15)
        q = np.maximum(targets[i], 1e-15)
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        # Compute JS divergence
        m = 0.5 * (p + q)
        js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        js_divergences.append(js_div)
    
    js_divergences = np.array(js_divergences)
    metrics['js_divergence'] = {
        'mean': js_divergences.mean(),
        'std': js_divergences.std(),
        'median': np.median(js_divergences)
    }
    
    return metrics


def compute_prediction_quality_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, Any]:
    """Compute prediction quality metrics like accuracy, F1 score, etc.
    
    Args:
        predictions: Predicted probability distributions
        targets: Target probability distributions
        
    Returns:
        Dictionary containing prediction quality metrics
    """
    if predictions.ndim == 3:
        # Average across fractions if 3D
        predictions = predictions.mean(axis=0)
        targets = targets.mean(axis=0)
    
    # Get predicted and true labels
    pred_labels = predictions.argmax(axis=1)
    true_labels = targets.argmax(axis=1)
    
    # Compute accuracy
    accuracy = (pred_labels == true_labels).mean()
    
    # Compute per-class accuracies
    n_classes = predictions.shape[1]
    per_class_accuracy = {}
    
    for class_idx in range(n_classes):
        class_mask = (true_labels == class_idx)
        if class_mask.sum() > 0:
            class_acc = (pred_labels[class_mask] == class_idx).mean()
            per_class_accuracy[f'class_{class_idx}'] = class_acc
        else:
            per_class_accuracy[f'class_{class_idx}'] = 0.0
    
    # Compute confidence statistics
    confidences = predictions.max(axis=1)
    confidence_stats = {
        'mean': confidences.mean(),
        'std': confidences.std(),
        'median': np.median(confidences),
        'min': confidences.min(),
        'max': confidences.max()
    }
    
    # Compute entropy statistics
    entropies = -np.sum(predictions * np.log(np.maximum(predictions, 1e-15)), axis=1)
    entropy_stats = {
        'mean': entropies.mean(),
        'std': entropies.std(),
        'median': np.median(entropies),
        'min': entropies.min(),
        'max': entropies.max()
    }
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'confidence_stats': confidence_stats,
        'entropy_stats': entropy_stats
    }