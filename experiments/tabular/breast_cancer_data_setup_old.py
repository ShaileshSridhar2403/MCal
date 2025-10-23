import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from typing import Tuple, Dict, Any, Optional
import xgboost as xgb
from pathlib import Path

def load_breast_cancer_dataset(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load and preprocess the Wisconsin Breast Cancer dataset.

    Returns:
        X_train, y_train, X_test, y_test: Training and testing splits
    """
    # Load the dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='diagnosis')  # 0 = malignant, 1 = benign

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Reset indices
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, y_train, X_test, y_test

def get_feature_groups() -> Dict[str, list]:
    """
    Define medically meaningful feature groups for breast cancer diagnosis.
    Groups based on measurement categories and diagnostic importance.
    """
    return {
        'mean_features': [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension'
        ],
        'se_features': [
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error'
        ],
        'worst_features': [
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ],
        'size_features': [
            'mean radius', 'mean perimeter', 'mean area',
            'radius error', 'perimeter error', 'area error',
            'worst radius', 'worst perimeter', 'worst area'
        ],
        'texture_features': [
            'mean texture', 'mean smoothness', 'mean compactness',
            'texture error', 'smoothness error', 'compactness error',
            'worst texture', 'worst smoothness', 'worst compactness'
        ],
        'shape_features': [
            'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
            'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
    }

def simulate_imaging_equipment_failure(X: pd.DataFrame, failure_rate: float = 0.3,
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Simulate imaging equipment failures that affect specific measurement categories.
    Different equipment may fail to capture certain types of measurements.
    """
    np.random.seed(random_state)
    X_ablated = X.copy()
    feature_groups = get_feature_groups()

    # Simulate different types of equipment failures
    equipment_failures = {
        'texture_analyzer': feature_groups['texture_features'],
        'shape_detector': feature_groups['shape_features'],
        'size_measurement': feature_groups['size_features'][:3],  # Only mean size features
    }

    n_samples = len(X)

    for equipment, affected_features in equipment_failures.items():
        # Each equipment has independent failure probability
        failure_mask = np.random.random(n_samples) < failure_rate
        available_features = [f for f in affected_features if f in X.columns]

        if available_features:
            X_ablated.loc[failure_mask, available_features] = np.nan

    return X_ablated

def simulate_biopsy_protocol_variation(X: pd.DataFrame, protocol_variation_rate: float = 0.4,
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Simulate variations in biopsy protocols that result in different measurement completeness.
    Some protocols may focus on specific diagnostic features.
    """
    np.random.seed(random_state + 1)
    X_ablated = X.copy()
    feature_groups = get_feature_groups()

    n_samples = len(X)

    # Different biopsy protocols with varying focus
    protocols = {
        'basic_protocol': feature_groups['se_features'],  # Skip error measurements
        'rapid_protocol': feature_groups['worst_features'],  # Skip worst-case measurements
        'limited_protocol': feature_groups['shape_features'][4:],  # Skip some shape features
    }

    for protocol, excluded_features in protocols.items():
        # Each protocol variation affects different samples
        protocol_mask = np.random.random(n_samples) < protocol_variation_rate
        available_features = [f for f in excluded_features if f in X.columns]

        if available_features:
            X_ablated.loc[protocol_mask, available_features] = np.nan

    return X_ablated

def simulate_measurement_precision_loss(X: pd.DataFrame, precision_loss_rate: float = 0.2,
                                      random_state: int = 42) -> pd.DataFrame:
    """
    Simulate loss of measurement precision where some features become unavailable
    due to calibration issues or measurement uncertainties.
    """
    np.random.seed(random_state + 2)
    X_ablated = X.copy()

    n_samples, n_features = X.shape

    # Create random missing pattern with some correlation structure
    missing_prob = np.random.beta(2, 5, n_features) * precision_loss_rate

    for i, col in enumerate(X.columns):
        missing_mask = np.random.random(n_samples) < missing_prob[i]
        X_ablated.loc[missing_mask, col] = np.nan

    return X_ablated

def apply_missing_ablation(X: pd.DataFrame, ablation_strategy: str = 'combined',
                          missing_fraction: float = 0.3, random_state: int = 42) -> pd.DataFrame:
    """
    Apply medical-realistic missing data ablation to breast cancer features.

    Args:
        X: Input features
        ablation_strategy: Type of ablation ('equipment', 'protocol', 'precision', 'combined', 'random')
        missing_fraction: Fraction of data to make missing (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        X_ablated: Features with missing data applied
    """
    if missing_fraction == 0.0:
        return X.copy()

    if ablation_strategy == 'equipment':
        return simulate_imaging_equipment_failure(X, missing_fraction, random_state)
    elif ablation_strategy == 'protocol':
        return simulate_biopsy_protocol_variation(X, missing_fraction, random_state)
    elif ablation_strategy == 'precision':
        return simulate_measurement_precision_loss(X, missing_fraction, random_state)
    elif ablation_strategy == 'combined':
        # Apply all strategies with reduced individual rates
        X_ablated = X.copy()
        individual_rate = missing_fraction / 3

        X_ablated = simulate_imaging_equipment_failure(X_ablated, individual_rate, random_state)
        X_ablated = simulate_biopsy_protocol_variation(X_ablated, individual_rate, random_state + 1)
        X_ablated = simulate_measurement_precision_loss(X_ablated, individual_rate, random_state + 2)

        return X_ablated
    elif ablation_strategy == 'random':
        # Random missing for comparison
        np.random.seed(random_state)
        X_ablated = X.copy()
        n_samples, n_features = X.shape

        missing_mask = np.random.random((n_samples, n_features)) < missing_fraction
        X_ablated = X_ablated.mask(missing_mask)

        return X_ablated
    else:
        raise ValueError(f"Unknown ablation strategy: {ablation_strategy}")

def prepare_data_for_training(X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            normalize: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare data for training by handling scaling and ensuring consistency.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        normalize: Whether to standardize features

    Returns:
        Prepared training and test data
    """
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    if normalize:
        # Fit scaler on training data (ignoring NaN values)
        scaler = StandardScaler()

        # Create mask for non-NaN values in training data
        train_mask = ~X_train_processed.isnull()

        # Fit scaler only on available training data
        train_values = X_train_processed.values
        train_values_flat = train_values[train_mask.values]

        if len(train_values_flat) > 0:
            # Fit on column-wise available data
            for col in X_train_processed.columns:
                col_data = X_train_processed[col].dropna()
                if len(col_data) > 0:
                    col_scaler = StandardScaler()
                    col_scaler.fit(col_data.values.reshape(-1, 1))

                    # Transform both train and test for this column
                    train_col_mask = ~X_train_processed[col].isnull()
                    test_col_mask = ~X_test_processed[col].isnull()

                    X_train_processed.loc[train_col_mask, col] = col_scaler.transform(
                        X_train_processed.loc[train_col_mask, col].values.reshape(-1, 1)
                    ).flatten()

                    X_test_processed.loc[test_col_mask, col] = col_scaler.transform(
                        X_test_processed.loc[test_col_mask, col].values.reshape(-1, 1)
                    ).flatten()

    return X_train_processed, y_train, X_test_processed, y_test

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                       model_params: Optional[Dict[str, Any]] = None,
                       random_state: int = 42) -> xgb.XGBClassifier:
    """
    Train XGBoost model on breast cancer data with missing value handling.

    Args:
        X_train: Training features (may contain NaN)
        y_train: Training labels
        model_params: XGBoost parameters
        random_state: Random seed

    Returns:
        Trained XGBoost model
    """
    if model_params is None:
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'tree_method': 'hist',
            'enable_categorical': False
        }

    # XGBoost handles NaN values natively
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    return model

def generate_missingness_levels(data_dir: str, n_levels: int = 10) -> None:
    """
    Generate missingness levels file for breast cancer dataset.

    Args:
        data_dir: Directory to save the file
        n_levels: Number of missingness levels to generate
    """
    os.makedirs(data_dir, exist_ok=True)

    # Create missingness levels from 0% to 90%
    missingness_levels = np.linspace(0.0, 0.9, n_levels)

    missingness_data = {
        'dataset': 'breast_cancer',
        'description': 'Wisconsin Breast Cancer Dataset missingness levels',
        'levels': missingness_levels.tolist(),
        'strategies': ['equipment', 'protocol', 'precision', 'combined', 'random']
    }

    filepath = os.path.join(data_dir, 'breast_cancer_missingness_levels.json')
    with open(filepath, 'w') as f:
        json.dump(missingness_data, f, indent=2)

    print(f"Generated missingness levels file: {filepath}")

def save_dataset_info(data_dir: str) -> None:
    """
    Save dataset information and statistics.
    """
    X_train, y_train, X_test, y_test = load_breast_cancer_dataset()

    dataset_info = {
        'name': 'Wisconsin Breast Cancer Dataset',
        'description': 'Binary classification of breast cancer diagnosis',
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'n_features': len(X_train.columns),
        'feature_names': X_train.columns.tolist(),
        'n_classes': 2,
        'class_names': ['malignant', 'benign'],
        'class_distribution_train': y_train.value_counts().to_dict(),
        'class_distribution_test': y_test.value_counts().to_dict(),
        'feature_groups': get_feature_groups()
    }

    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'breast_cancer_dataset_info.json')
    with open(filepath, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Saved dataset info: {filepath}")

if __name__ == "__main__":
    # Example usage and testing
    data_dir = "/home/antonxue/shailesh/MCal/data/tabular"

    print("Loading Breast Cancer Wisconsin dataset...")
    X_train, y_train, X_test, y_test = load_breast_cancer_dataset()

    print(f"Dataset shape: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")

    # Test ablation strategies
    print("\nTesting ablation strategies...")
    for strategy in ['equipment', 'protocol', 'precision', 'combined', 'random']:
        X_ablated = apply_missing_ablation(X_train, strategy, 0.3, 42)
        missing_rate = X_ablated.isnull().sum().sum() / (X_ablated.shape[0] * X_ablated.shape[1])
        print(f"{strategy}: {missing_rate:.3f} missing rate")

    # Test model training
    print("\nTesting model training...")
    X_ablated = apply_missing_ablation(X_train, 'combined', 0.2, 42)
    X_train_prep, y_train_prep, X_test_prep, y_test_prep = prepare_data_for_training(
        X_ablated, y_train, X_test, y_test
    )

    model = train_xgboost_model(X_train_prep, y_train_prep)
    train_accuracy = model.score(X_train_prep, y_train_prep)
    test_accuracy = model.score(X_test_prep, y_test_prep)
    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")

    # Generate files
    generate_missingness_levels(data_dir)
    save_dataset_info(data_dir)

    print("Breast cancer data setup completed successfully!")