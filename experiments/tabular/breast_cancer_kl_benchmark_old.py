import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import XGBoost and sklearn
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Import MCal modules
mcal_root = current_dir.parent.parent
sys.path.insert(0, str(mcal_root))
sys.path.insert(0, str(mcal_root / "src"))

try:
    from calibrators.mcal import MCal
    from calibrators.mcal_ce import MCal_CE
except ImportError as e:
    print(f"Warning: Could not import MCal modules: {e}")
    MCal = None
    MCal_CE = None

# Import local data setup
from breast_cancer_data_setup import (
    load_breast_cancer_dataset,
    apply_missing_ablation,
    prepare_data_for_training,
    train_xgboost_model
)

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate KL divergence between two probability distributions.

    Args:
        p: True distribution
        q: Predicted distribution
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence
    """
    # Clip probabilities to avoid log(0)
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)

    # Ensure probabilities sum to 1
    p = p / np.sum(p, axis=1, keepdims=True)
    q = q / np.sum(q, axis=1, keepdims=True)

    return np.mean(np.sum(p * np.log(p / q), axis=1))

def prepare_binary_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Convert binary classifier output to proper probability distribution.

    Args:
        probs: Raw probabilities from binary classifier

    Returns:
        2D probability array with shape (n_samples, 2)
    """
    if probs.ndim == 1:
        # Convert 1D probabilities to 2D [P(class=0), P(class=1)]
        probs_2d = np.column_stack([1 - probs, probs])
    else:
        probs_2d = probs

    # Ensure probabilities are properly normalized
    probs_2d = np.clip(probs_2d, 1e-10, 1 - 1e-10)
    row_sums = probs_2d.sum(axis=1, keepdims=True)
    probs_2d = probs_2d / row_sums

    return probs_2d

def create_uniform_distribution(n_samples: int, n_classes: int = 2) -> np.ndarray:
    """Create uniform distribution for KL divergence comparison."""
    return np.full((n_samples, n_classes), 1.0 / n_classes)

def apply_temperature_scaling(logits_test: np.ndarray, logits_val: np.ndarray, y_val: np.ndarray,
                            max_iter: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Apply temperature scaling calibration.

    Args:
        logits_test: Test logits to calibrate
        logits_val: Validation logits for temperature fitting
        y_val: Validation labels for temperature fitting
        max_iter: Maximum iterations for optimization

    Returns:
        Calibrated probabilities and optimal temperature
    """
    from scipy.optimize import minimize_scalar

    # Convert logits to log-odds for binary classification
    if logits_val.ndim == 1:
        # Binary case: convert to 2D
        logits_val_2d = np.column_stack([-logits_val, logits_val])
    else:
        logits_val_2d = logits_val

    if logits_test.ndim == 1:
        # Binary case: convert to 2D
        logits_test_2d = np.column_stack([-logits_test, logits_test])
    else:
        logits_test_2d = logits_test

    def temperature_loss(temp):
        if temp <= 0:
            return 1e6
        try:
            scaled_logits = logits_val_2d / temp
            probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
            return log_loss(y_val, probs)
        except:
            return 1e6

    # Find optimal temperature
    result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
    optimal_temp = result.x

    # Apply optimal temperature to test logits
    scaled_logits = logits_test_2d / optimal_temp
    calibrated_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)

    return calibrated_probs, optimal_temp

def apply_platt_scaling(y_scores: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    Apply Platt scaling calibration.

    Args:
        y_scores: Raw scores from model
        y_val: Validation labels

    Returns:
        Calibrated probabilities
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import BaseEstimator, ClassifierMixin

    class ScoreClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, scores):
            self.scores = scores
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            return self

        def decision_function(self, X):
            return self.scores[X]

    # Create a dummy classifier with the scores
    dummy_clf = ScoreClassifier(y_scores)

    # Apply Platt scaling
    calibrated_clf = CalibratedClassifierCV(dummy_clf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(np.arange(len(y_val)).reshape(-1, 1), y_val)

    # Get calibrated probabilities
    calibrated_probs = calibrated_clf.predict_proba(np.arange(len(y_scores)).reshape(-1, 1))

    return calibrated_probs

def apply_mcal_calibrator(X_val: np.ndarray, y_val: np.ndarray,
                         X_test: np.ndarray, logits_val: np.ndarray,
                         logits_test: np.ndarray) -> np.ndarray:
    """
    Apply MCal vector scaling calibration.

    Args:
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        logits_val: Validation logits
        logits_test: Test logits

    Returns:
        Calibrated test probabilities
    """
    if MCal is None:
        raise ImportError("MCal not available")

    try:
        # Convert binary labels to one-hot for MCal
        n_classes = 2
        y_val_onehot = np.eye(n_classes)[y_val.astype(int)]

        # Prepare logits as 2D arrays
        if logits_val.ndim == 1:
            logits_val_2d = np.column_stack([-logits_val, logits_val])
        else:
            logits_val_2d = logits_val

        if logits_test.ndim == 1:
            logits_test_2d = np.column_stack([-logits_test, logits_test])
        else:
            logits_test_2d = logits_test

        # Convert to torch tensors
        import torch
        logits_val_tensor = torch.tensor(logits_val_2d, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_onehot, dtype=torch.float32)
        logits_test_tensor = torch.tensor(logits_test_2d, dtype=torch.float32)

        # Initialize and fit MCal
        mcal_calibrator = MCal(num_classes=n_classes)
        mcal_calibrator.fit(logits_val_tensor, y_val_tensor)

        # Get calibrated probabilities
        calibrated_probs = mcal_calibrator.predict_proba(logits_test_tensor)

        # Convert back to numpy
        if hasattr(calibrated_probs, 'detach'):
            calibrated_probs = calibrated_probs.detach().cpu().numpy()

        return calibrated_probs

    except Exception as e:
        print(f"MCal calibration failed: {e}. Using temperature scaling fallback.")
        return apply_temperature_scaling(logits_test, logits_val, y_val)[0]

def apply_mcal_ce_calibrator(X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, logits_val: np.ndarray,
                           logits_test: np.ndarray) -> np.ndarray:
    """
    Apply MCal cross-entropy calibration.

    Args:
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        logits_val: Validation logits
        logits_test: Test logits

    Returns:
        Calibrated test probabilities
    """
    if MCal_CE is None:
        raise ImportError("MCal cross-entropy not available")

    try:
        # Convert binary labels to one-hot for MCal
        n_classes = 2
        y_val_onehot = np.eye(n_classes)[y_val.astype(int)]

        # Prepare logits as 2D arrays
        if logits_val.ndim == 1:
            logits_val_2d = np.column_stack([-logits_val, logits_val])
        else:
            logits_val_2d = logits_val

        if logits_test.ndim == 1:
            logits_test_2d = np.column_stack([-logits_test, logits_test])
        else:
            logits_test_2d = logits_test

        # Convert to torch tensors
        import torch
        logits_val_tensor = torch.tensor(logits_val_2d, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_onehot, dtype=torch.float32)
        logits_test_tensor = torch.tensor(logits_test_2d, dtype=torch.float32)

        # Initialize and fit MCal CE
        mcal_ce_calibrator = MCal_CE(num_classes=n_classes)
        mcal_ce_calibrator.fit(logits_val_tensor, y_val_tensor)

        # Get calibrated probabilities
        calibrated_probs = mcal_ce_calibrator.predict_proba(logits_test_tensor)

        # Convert back to numpy
        if hasattr(calibrated_probs, 'detach'):
            calibrated_probs = calibrated_probs.detach().cpu().numpy()

        return calibrated_probs

    except Exception as e:
        print(f"MCal CE calibration failed: {e}. Using temperature scaling fallback.")
        return apply_temperature_scaling(logits_test, logits_val, y_val)[0]

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        random_state: int = 42) -> xgb.XGBClassifier:
    """Train baseline XGBoost model on complete training data."""
    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': random_state,
        'tree_method': 'hist'
    }

    model = train_xgboost_model(X_train, y_train, model_params, random_state)
    return model

def train_retrain_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       missing_fraction: float = 0.3,
                       ablation_strategy: str = 'combined',
                       random_state: int = 42) -> xgb.XGBClassifier:
    """Train XGBoost model on missing data for retrain method."""

    # Apply missing data ablation to training set
    X_train_missing = apply_missing_ablation(
        X_train, ablation_strategy, missing_fraction, random_state
    )

    # Prepare data
    X_train_prep, y_train_prep, X_val_prep, y_val_prep = prepare_data_for_training(
        X_train_missing, y_train, X_val, y_val, normalize=True
    )

    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': random_state,
        'tree_method': 'hist'
    }

    model = train_xgboost_model(X_train_prep, y_train_prep, model_params, random_state)
    return model

def run_calibration_method(method: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          missing_fraction: float = 0.0,
                          ablation_strategy: str = 'combined',
                          random_state: int = 42) -> Dict[str, Any]:
    """
    Run a specific calibration method and return results.

    Args:
        method: Calibration method name
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        missing_fraction: Fraction of missing data to simulate
        ablation_strategy: Strategy for creating missing data
        random_state: Random seed

    Returns:
        Dictionary containing results
    """
    print(f"Running {method} with {missing_fraction:.1%} missing data...")

    # Apply missing data ablation to test set
    X_test_ablated = apply_missing_ablation(
        X_test, ablation_strategy, missing_fraction, random_state
    )

    # Prepare data
    X_train_prep, y_train_prep, X_val_prep, y_val_prep = prepare_data_for_training(
        X_train, y_train, X_val, y_val, normalize=True
    )

    X_test_prep = X_test_ablated.copy()

    # Normalize test data using training statistics
    for col in X_test_prep.columns:
        if col in X_train_prep.columns:
            train_col = X_train_prep[col].dropna()
            if len(train_col) > 0:
                mean_val = train_col.mean()
                std_val = train_col.std()
                if std_val > 0:
                    test_mask = ~X_test_prep[col].isnull()
                    X_test_prep.loc[test_mask, col] = (X_test_prep.loc[test_mask, col] - mean_val) / std_val

    # Train appropriate model based on method
    if method == 'retrain':
        model = train_retrain_model(X_train, y_train, X_val, y_val,
                                  missing_fraction, ablation_strategy, random_state)
        # For retrain, we need to use the same missing pattern on validation
        X_val_prep = apply_missing_ablation(X_val, ablation_strategy, missing_fraction, random_state + 1)
        # Normalize validation data
        for col in X_val_prep.columns:
            if col in X_train_prep.columns:
                train_col = X_train_prep[col].dropna()
                if len(train_col) > 0:
                    mean_val = train_col.mean()
                    std_val = train_col.std()
                    if std_val > 0:
                        val_mask = ~X_val_prep[col].isnull()
                        X_val_prep.loc[val_mask, col] = (X_val_prep.loc[val_mask, col] - mean_val) / std_val
    else:
        model = train_baseline_model(X_train, y_train, X_val, y_val, random_state)

    # Get predictions and logits
    y_pred_test = model.predict(X_test_prep)
    y_proba_test = model.predict_proba(X_test_prep)

    # Get validation predictions for calibration
    y_proba_val = model.predict_proba(X_val_prep)

    # Convert to proper probability format
    proba_test = prepare_binary_probabilities(y_proba_test[:, 1] if y_proba_test.ndim > 1 else y_proba_test)
    proba_val = prepare_binary_probabilities(y_proba_val[:, 1] if y_proba_val.ndim > 1 else y_proba_val)

    # Get logits (inverse sigmoid of probabilities)
    def prob_to_logit(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))

    logits_test = prob_to_logit(proba_test[:, 1])
    logits_val = prob_to_logit(proba_val[:, 1])

    # Apply calibration based on method
    if method == 'baseline' or method == 'retrain':
        calibrated_probs = proba_test
    elif method == 'mcal':
        calibrated_probs = apply_mcal_calibrator(
            X_val_prep.values, y_val, X_test_prep.values, logits_val, logits_test
        )
    elif method == 'mcal_ce':
        calibrated_probs = apply_mcal_ce_calibrator(
            X_val_prep.values, y_val, X_test_prep.values, logits_val, logits_test
        )
    elif method == 'temperature':
        calibrated_probs, temp = apply_temperature_scaling(logits_test, logits_val, y_val)
    elif method == 'platt':
        calibrated_probs = apply_platt_scaling(logits_test, y_val)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure proper probability format
    calibrated_probs = prepare_binary_probabilities(
        calibrated_probs[:, 1] if calibrated_probs.ndim > 1 else calibrated_probs
    )

    # Calculate metrics
    accuracy = accuracy_score(y_test, (calibrated_probs[:, 1] > 0.5).astype(int))

    # Create ground truth distribution (one-hot encoded)
    y_test_onehot = np.eye(2)[y_test.astype(int)]

    # Calculate KL divergences
    kl_prob = calculate_kl_divergence(y_test_onehot, calibrated_probs)

    # Argmax probabilities (convert to hard predictions then back to probabilities)
    argmax_preds = np.argmax(calibrated_probs, axis=1)
    argmax_probs = np.eye(2)[argmax_preds]
    kl_argmax = calculate_kl_divergence(y_test_onehot, argmax_probs)

    return {
        'method': method,
        'missing_fraction': missing_fraction,
        'accuracy': accuracy,
        'kl_prob': kl_prob,
        'kl_argmax': kl_argmax,
        'calibrated_probabilities': calibrated_probs,
        'predictions': argmax_preds
    }

def run_breast_cancer_kl_benchmark(missing_fractions: List[float] = None,
                                  methods: List[str] = None,
                                  ablation_strategy: str = 'combined',
                                  random_state: int = 42,
                                  output_dir: str = "results") -> Dict[str, Any]:
    """
    Run comprehensive KL divergence benchmark on Breast Cancer Wisconsin dataset.

    Args:
        missing_fractions: List of missing data fractions to test
        methods: List of calibration methods to test
        ablation_strategy: Strategy for missing data simulation
        random_state: Random seed for reproducibility
        output_dir: Directory to save results

    Returns:
        Comprehensive results dictionary
    """
    if missing_fractions is None:
        missing_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if methods is None:
        methods = ['baseline', 'mcal', 'mcal_ce', 'temperature', 'platt', 'retrain']

    print("Loading Breast Cancer Wisconsin dataset...")
    X_train, y_train, X_test, y_test = load_breast_cancer_dataset(random_state=random_state)

    # Create validation split from training data
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )

    print(f"Dataset splits: {len(X_train_split)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"Testing {len(methods)} methods with {len(missing_fractions)} missing fractions")

    # Store all results
    all_results = []
    method_summaries = {}

    # Run experiments
    for method in methods:
        print(f"\n=== Testing {method.upper()} ===")
        method_results = []

        for missing_frac in missing_fractions:
            try:
                result = run_calibration_method(
                    method, X_train_split, y_train_split, X_val, y_val, X_test, y_test,
                    missing_frac, ablation_strategy, random_state
                )
                method_results.append(result)
                all_results.append(result)

                print(f"  {missing_frac:.1%} missing: "
                      f"Acc={result['accuracy']:.3f}, "
                      f"KL_prob={result['kl_prob']:.4f}, "
                      f"KL_argmax={result['kl_argmax']:.4f}")

            except Exception as e:
                print(f"  {missing_frac:.1%} missing: FAILED - {e}")
                continue

        # Calculate method summary statistics
        if method_results:
            accuracies = [r['accuracy'] for r in method_results]
            kl_probs = [r['kl_prob'] for r in method_results]
            kl_argmaxs = [r['kl_argmax'] for r in method_results]

            method_summaries[method] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'kl_prob_mean': np.mean(kl_probs),
                'kl_prob_std': np.std(kl_probs),
                'kl_argmax_mean': np.mean(kl_argmaxs),
                'kl_argmax_std': np.std(kl_argmaxs),
                'results': method_results
            }

    # Create comprehensive results
    comprehensive_results = {
        'dataset': 'breast_cancer_wisconsin',
        'ablation_strategy': ablation_strategy,
        'missing_fractions': missing_fractions,
        'methods': methods,
        'method_summaries': method_summaries,
        'all_results': all_results,
        'metadata': {
            'n_train': len(X_train_split),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': X_test.shape[1],
            'random_state': random_state
        }
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/json", exist_ok=True)

    # Save detailed results
    with open(f"{output_dir}/json/breast_cancer_detailed_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        json_results = convert_numpy(comprehensive_results)
        json.dump(json_results, f, indent=2)

    # Save aggregated results
    aggregated_results = {method: summary for method, summary in method_summaries.items()
                         if 'results' in summary}

    # Remove detailed results for cleaner aggregated file
    for method_data in aggregated_results.values():
        method_data.pop('results', None)

    with open(f"{output_dir}/json/aggregated_results_breast_cancer.json", 'w') as f:
        json.dump(convert_numpy(aggregated_results), f, indent=2)

    # Generate summary table
    generate_summary_table(method_summaries, output_dir)

    print(f"\n=== BREAST CANCER WISCONSIN BENCHMARK COMPLETE ===")
    print(f"Results saved to {output_dir}/")

    return comprehensive_results

def generate_summary_table(method_summaries: Dict[str, Any], output_dir: str) -> None:
    """Generate formatted summary table of results."""

    # Create table data
    table_lines = []
    table_lines.append("Breast Cancer Wisconsin Binary KL Divergence Comparison:")
    table_lines.append("+----------------------+---------------------+---------------------+")
    table_lines.append("| Method               | Average KL (Prob)  | Average KL (Argmax)|")
    table_lines.append("+======================+=====================+=====================+")

    # Sort methods for consistent output
    method_order = ['baseline', 'mcal', 'mcal_ce', 'temperature', 'platt', 'retrain']
    sorted_methods = [(m, method_summaries[m]) for m in method_order if m in method_summaries]

    for method, summary in sorted_methods:
        method_display = {
            'baseline': 'Baseline',
            'mcal': 'MCal Binary',
            'mcal_ce': 'MCal_CE Binary',
            'temperature': 'Temperature',
            'platt': 'Platt Scaling',
            'retrain': 'Retrain'
        }.get(method, method.title())

        kl_prob_mean = summary['kl_prob_mean']
        kl_prob_std = summary['kl_prob_std']
        kl_argmax_mean = summary['kl_argmax_mean']
        kl_argmax_std = summary['kl_argmax_std']

        table_lines.append(
            f"| {method_display:<20} | "
            f"{kl_prob_mean:.2e} ± {kl_prob_std:.2e} | "
            f"{kl_argmax_mean:.2e} ± {kl_argmax_std:.2e} |"
        )

    table_lines.append("+----------------------+---------------------+---------------------+")

    # Save table
    table_content = "\n".join(table_lines)

    with open(f"{output_dir}/kl_comparison_table_breast_cancer.txt", 'w') as f:
        f.write(table_content)

    print("\n" + table_content)

def main():
    """Main function for running the benchmark."""
    parser = argparse.ArgumentParser(description='Run Breast Cancer Wisconsin KL Divergence Benchmark')
    parser.add_argument('--missing-fractions', nargs='+', type=float,
                       default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       help='List of missing data fractions to test')
    parser.add_argument('--methods', nargs='+', type=str,
                       default=['baseline', 'mcal', 'mcal_ce', 'temperature', 'platt', 'retrain'],
                       help='List of calibration methods to test')
    parser.add_argument('--ablation-strategy', type=str, default='combined',
                       choices=['equipment', 'protocol', 'precision', 'combined', 'random'],
                       help='Strategy for missing data simulation')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Run benchmark
    results = run_breast_cancer_kl_benchmark(
        missing_fractions=args.missing_fractions,
        methods=args.methods,
        ablation_strategy=args.ablation_strategy,
        random_state=args.random_state,
        output_dir=args.output_dir
    )

    print(f"\nBenchmark completed successfully!")
    print(f"Results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()