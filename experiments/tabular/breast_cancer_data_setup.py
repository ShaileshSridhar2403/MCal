#!/usr/bin/env python3
"""
Simplified Breast Cancer Data Setup for MCal Tabular Benchmarks

This module provides a clean, simplified data loading interface for Breast Cancer
following the PhysioNet data setup pattern. Focuses on fractionwise MCAR
missing data ablation with minimal preprocessing.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from tqdm import tqdm
import xgboost as xgb

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent


def preprocess_breast_cancer_features(X, y):
    """
    Minimal preprocessing for breast cancer dataset.
    (No StandardScaler - keep raw features like PhysioNet)

    Args:
        X: Feature array from sklearn
        y: Labels array from sklearn

    Returns:
        pd.DataFrame, pd.Series: Preprocessed features and labels
    """
    # Convert to DataFrame for compatibility
    data = load_breast_cancer()
    feature_names = data.feature_names

    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='diagnosis')

    print(f"Preprocessed breast cancer features: {X_df.shape[1]} features")
    print(f"Features: {list(X_df.columns)}")
    print(f"Label distribution: {y_series.value_counts().to_dict()}")

    return X_df, y_series


def balance_breast_cancer_dataset(X, y):
    """
    Balance breast cancer dataset by undersampling majority class.
    (Adapted from PhysioNet's balance_physionet_dataset for binary)
    """
    # Combine for easier manipulation
    data = pd.concat([X, y], axis=1)

    # Get class counts
    class_counts = y.value_counts().sort_index()
    print(f"Original class distribution:")
    print(f"  Malignant (0): {class_counts.get(0, 0)}")
    print(f"  Benign (1): {class_counts.get(1, 0)}")

    # Undersample majority class to match minority class
    min_count = class_counts.min()

    balanced_dfs = []
    for class_val in sorted(y.unique()):
        class_data = data[data['diagnosis'] == class_val]
        if len(class_data) > min_count:
            class_data = resample(class_data, replace=False, n_samples=min_count, random_state=42)
        balanced_dfs.append(class_data)

    # Combine and shuffle
    data_balanced = pd.concat(balanced_dfs, ignore_index=True)
    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split back into X and y
    X_balanced = data_balanced.drop('diagnosis', axis=1)
    y_balanced = data_balanced['diagnosis']

    balanced_counts = y_balanced.value_counts().sort_index()
    print(f"Balanced class distribution:")
    print(f"  Malignant (0): {balanced_counts.get(0, 0)}")
    print(f"  Benign (1): {balanced_counts.get(1, 0)}")

    return X_balanced, y_balanced


def load_clean_breast_cancer_data(n_samples=1000):
    """Load and balance breast cancer dataset using sklearn"""
    # Load breast cancer dataset from sklearn
    data = load_breast_cancer()
    X, y = data.data, data.target

    print(f"Loaded breast cancer dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {data.target_names}")

    # Apply minimal preprocessing
    X_df, y_series = preprocess_breast_cancer_features(X, y)

    # Train/test split with stratification (use train split for benchmark)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, stratify=y_series, random_state=42
    )

    print(f"Train/test split: {len(X_train)} train, {len(X_test)} test samples")

    # Apply simple balancing (undersample majority class)
    X_balanced, y_balanced = balance_breast_cancer_dataset(X_train, y_train)

    # Sample if needed
    if n_samples < len(X_balanced):
        combined = pd.concat([X_balanced, y_balanced], axis=1)
        sampled = combined.sample(n=n_samples, random_state=42).reset_index(drop=True)
        X_balanced = sampled.drop('diagnosis', axis=1)
        y_balanced = sampled['diagnosis']
        print(f"Sampled {n_samples} samples from balanced dataset")

    return X_balanced, y_balanced


def train_xgboost_model(X_train, y_train, missing_value=None):
    """
    Train XGBoost model with optional custom missing value handling.
    (Simplified version following PhysioNet pattern)

    Args:
        X_train: Training features
        y_train: Training labels
        missing_value: Value to treat as missing (e.g., -10, np.nan)

    Returns:
        Trained XGBoost model
    """
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',        # Binary log loss
        'eta': 0.1,
        'max_depth': 6,
        'seed': 42,
        'tree_method': 'hist',
        'enable_categorical': False
    }

    # Add missing value parameter if specified
    if missing_value is not None:
        model = xgb.XGBClassifier(missing=missing_value, **params)
        print(f"Trained XGBoost model on {len(X_train)} samples with missing={missing_value}")
    else:
        model = xgb.XGBClassifier(**params)
        print(f"Trained XGBoost model on {len(X_train)} samples")

    model.fit(X_train, y_train)
    return model


def apply_missing_data_simulation(data, removal_fraction):
    """
    Apply MCAR missing data - equivalent to PatchCutout for tabular.
    (Identical to PhysioNet version)

    Args:
        data: DataFrame or array with complete features
        removal_fraction: Float between 0-1, fraction of features to remove per sample

    Returns:
        DataFrame/array with randomly missing values

    Example:
        - Input: 100 samples x 30 features (all complete)
        - removal_fraction = 0.3
        - Output: 100 samples x 30 features (30% randomly set to NaN per sample)
    """
    data_missing = data.copy()
    n_samples, n_features = data.shape

    for i in range(n_samples):
        # For each sample, randomly select features to remove
        n_remove = int(n_features * removal_fraction)
        if n_remove > 0:
            features_to_remove = np.random.choice(n_features, size=n_remove, replace=False)
            data_missing.iloc[i, features_to_remove] = np.nan

    return data_missing


def load_breast_cancer_data(model_type="vanilla", fill_value="mean", n_samples=1000, n_fractions=10,
                           data_source="sklearn", missing_value=None):
    """
    Simple, clean breast cancer data loading following PhysioNet pattern.

    Args:
        model_type: "vanilla" or "retrained" - controls training data selection
        fill_value: "mean", "nan", "zero", or "-10" - how to handle missing values during prediction
        n_samples: Number of samples to use
        n_fractions: Number of ablation fractions
        data_source: "sklearn" - indicates sklearn data source
        missing_value: Value to treat as missing in XGBoost (e.g., -10, np.nan)

    Returns:
        (predictions, labels): Torch tensors with shapes (k, n, c) and (n,)
        where c=2 for breast cancer (binary classification)
    """
    print(f"Loading breast cancer data with {n_samples} samples, {n_fractions} fractions...")
    print(f"Model type: {model_type}, Fill value: {fill_value}")

    # 1. Load clean base data
    X_clean, y_clean = load_clean_breast_cancer_data(n_samples)

    # 2. Prepare training data based on model_type - INLINE LOGIC
    if model_type == "vanilla":
        # Train on clean data
        X_train = X_clean.copy()

    elif model_type == "retrained":
        # Train on data with binomial missingness (50% probability per feature per row)
        X_train = X_clean.copy()
        n_samples_train, n_features = X_train.shape

        # Apply binomial missingness: each feature has 50% chance of being missing (coin flip)
        for i in range(n_samples_train):
            for j in range(n_features):
                if np.random.random() < 0.5:  # Coin flip: 50% probability
                    X_train.iloc[i, j] = np.nan

    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    # Impute for training (XGBoost needs complete data for training)
    imputer = SimpleImputer(strategy='mean')  # Default is mean imputation
    X_train = imputer.fit_transform(X_train)

    # 3. Train model
    model = train_xgboost_model(X_train, y_clean, missing_value=missing_value)

    # 4. Generate predictions across ablation fractions
    ablation_fractions = [i/n_fractions for i in range(n_fractions)]
    all_probs = []

    print(f"Generating predictions across {n_fractions} ablation fractions...")

    for fraction in tqdm(ablation_fractions, desc="Ablation fractions"):
        # Apply missing data simulation
        data_with_missing = apply_missing_data_simulation(X_clean, fraction)

        # Apply preprocessing based on fill_value - INLINE CODE
        if fill_value == "mean":
            # Mean imputation (no scaling needed for XGBoost)
            processed_data = imputer.transform(data_with_missing)
        elif fill_value == "zero":
            # Zero-fill (no scaling needed for XGBoost)
            processed_data = data_with_missing.fillna(0).values
        elif fill_value == "nan":
            # Keep NaNs for XGBoost native missing value handling
            processed_data = data_with_missing.values  # Keep as-is
        elif fill_value == "-10":
            # Fill with -10 for custom XGBoost missing value handling
            processed_data = data_with_missing.fillna(-10).values
        else:
            raise ValueError(f"Invalid fill_value: {fill_value}")

        # Get predictions
        predictions = model.predict_proba(processed_data)
        all_probs.append(predictions)

    # Convert to torch tensors
    predictions_tensor = torch.stack([torch.from_numpy(p.astype(np.float32)) for p in all_probs])
    labels_tensor = torch.from_numpy(y_clean.values).long()

    print(f"Generated predictions shape: {predictions_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")

    return predictions_tensor, labels_tensor


def test_breast_cancer_loading():
    """Test breast cancer data loading functionality."""
    print("Testing simplified breast cancer data loading...")

    try:
        # Test basic data loading
        predictions, labels = load_breast_cancer_data(
            model_type="vanilla",
            fill_value="mean",
            n_samples=100,  # Small sample for testing
            n_fractions=5
        )

        print(f"✓ Successfully loaded data:")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Predictions type: {type(predictions)}")
        print(f"  Labels type: {type(labels)}")

        # Verify data properties
        assert predictions.shape[0] == 5, "Incorrect number of fractions"
        assert predictions.shape[1] == labels.shape[0], "Sample count mismatch"
        assert predictions.shape[2] == 2, "Should be binary classification"

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the simplified data loading functionality
    test_breast_cancer_loading()