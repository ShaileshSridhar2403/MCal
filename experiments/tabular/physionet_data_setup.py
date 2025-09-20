#!/usr/bin/env python3
"""
PhysioNet Data Setup for MCal Tabular Benchmarks

This module provides data loading, preprocessing, and missing data simulation
for the PhysioNet Challenge 2012 dataset following MCal benchmark patterns.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm
import xgboost as xgb

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))
sys.path.insert(0, str(mcal_root / "src"))

# Add XAI_Benchmark to path for data utilities (only when needed)
try:
    xai_root = mcal_root.parent / "XAI-Benchmark"
    sys.path.append(str(xai_root / "tabular"))
    from physionet_utils import (
        load_and_create_missingness_train,
        load_and_create_missingness_test,
        load_and_create_missingness_full
    )
except ImportError:
    print("Warning: Could not import physionet_utils from XAI_Benchmark")
    print("Make sure XAI_Benchmark is available in the parent directory")
    # Define stub functions for demo purposes
    def load_and_create_missingness_train(missingness_dir):
        return None
    def load_and_create_missingness_test(missingness_dir):
        return None
    def load_and_create_missingness_full(missingness_dir):
        return None


def randomly_remove_features(X, fraction):
    """
    Randomly remove a fraction of features from each sample.

    Args:
        X (pd.DataFrame or np.ndarray): Input features
        fraction (float): Fraction of features to remove (0 to 1)

    Returns:
        X_removed: Data with randomly removed features
    """
    if isinstance(X, pd.DataFrame):
        X_removed = X.copy()
        for column in X_removed.columns:
            mask = np.random.random(len(X_removed)) < fraction
            X_removed.loc[mask, column] = np.nan
    else:
        X_removed = X.copy()
        n_samples, n_features = X.shape
        for i in range(n_samples):
            n_remove = int(n_features * fraction)
            if n_remove > 0:
                features_to_remove = np.random.choice(n_features, size=n_remove, replace=False)
                X_removed[i, features_to_remove] = np.nan

    return X_removed


def balance_physionet_dataset(df, target_column='In-hospital_death'):
    """
    Balance PhysioNet dataset by undersampling majority class.

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Name of the target column

    Returns:
        pd.DataFrame: Balanced dataset
    """
    # Separate majority and minority classes
    df_majority = df[df[target_column] == 0]  # Survival (majority)
    df_minority = df[df[target_column] == 1]  # Death (minority)

    print(f"Original class distribution:")
    print(f"  Survival (0): {len(df_majority)}")
    print(f"  Death (1): {len(df_minority)}")

    # Undersample majority class to match minority class
    df_majority_undersampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    # Combine minority class with undersampled majority class
    df_balanced = pd.concat([df_majority_undersampled, df_minority])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced class distribution:")
    print(f"  Survival (0): {len(df_balanced[df_balanced[target_column] == 0])}")
    print(f"  Death (1): {len(df_balanced[df_balanced[target_column] == 1])}")

    return df_balanced


def load_or_create_balanced_physionet_dataset(
    missingness_dir="missingness_levels",
    balanced_file_path="balanced_physionet_dataset.csv",
    missingness_range="0-30",
    overwrite=False
):
    """
    Load balanced PhysioNet dataset if it exists, or create and save one.

    Args:
        missingness_dir (str): Directory containing missingness level files
        balanced_file_path (str): Path to save/load balanced dataset
        missingness_range (str): "0-30", "30-100", or "full"
        overwrite (bool): Whether to overwrite existing dataset

    Returns:
        pd.DataFrame: Balanced PhysioNet dataset
    """
    balanced_path = Path(balanced_file_path)

    if not overwrite and balanced_path.exists():
        print(f"Loading existing balanced PhysioNet dataset from {balanced_path}")
        return pd.read_csv(balanced_path)

    print(f"Creating balanced PhysioNet dataset (missingness range: {missingness_range})")

    # Load data based on missingness range
    if missingness_range == "0-30":
        df = load_and_create_missingness_train(missingness_dir)
    elif missingness_range == "30-100":
        df = load_and_create_missingness_test(missingness_dir)
    elif missingness_range == "full":
        df = load_and_create_missingness_full(missingness_dir)
    else:
        raise ValueError(f"Invalid missingness_range: {missingness_range}")

    if df is None:
        raise ValueError("Failed to load PhysioNet data")

    # Balance the dataset
    df_balanced = balance_physionet_dataset(df, 'In-hospital_death')

    # Save balanced dataset
    balanced_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(balanced_path, index=False)
    print(f"Balanced dataset saved to {balanced_path}")

    return df_balanced


def generate_fractionwise_predictions(
    model, X, y, removal_fractions=None,
    imputation_strategy="mean", batch_size=None
):
    """
    Generate fractionwise predictions for different missing data levels.

    Args:
        model: Trained XGBoost model
        X (pd.DataFrame): Feature data
        y (pd.Series): Target labels
        removal_fractions (list): Fractions of features to remove
        imputation_strategy (str): "mean", "zero", or "xgboost_native"
        batch_size (int): Batch size for processing (unused for XGBoost)

    Returns:
        np.ndarray: Predictions array of shape (n_fractions, n_samples, n_classes)
    """
    if removal_fractions is None:
        removal_fractions = np.linspace(0, 0.9, 10)

    n_samples = len(X)
    n_classes = 2  # Binary classification for PhysioNet
    n_fractions = len(removal_fractions)

    # Initialize predictions array
    predictions = np.zeros((n_fractions, n_samples, n_classes))

    # Initialize preprocessing
    if imputation_strategy == "mean":
        imputer = SimpleImputer(strategy='mean')
        X_processed = imputer.fit_transform(X)
    elif imputation_strategy == "zero":
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_processed = imputer.fit_transform(X)
    elif imputation_strategy == "xgboost_native":
        imputer = None
        X_processed = X.values
    else:
        raise ValueError(f"Invalid imputation_strategy: {imputation_strategy}")

    # Fit scaler on processed data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    for i, fraction in enumerate(tqdm(removal_fractions, desc="Generating fractionwise predictions")):
        # Apply missing data
        if fraction == 0:
            X_eval = X.copy()
        else:
            X_eval = randomly_remove_features(X, fraction)

        # Apply preprocessing
        if imputation_strategy in ["mean", "zero"]:
            X_eval_processed = imputer.transform(X_eval)
            X_eval_scaled = scaler.transform(X_eval_processed)
        elif imputation_strategy == "xgboost_native":
            X_eval_scaled = scaler.transform(X_eval)

        # Generate predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_eval_scaled)
        else:
            # For XGBoost DMatrix
            dtest = xgb.DMatrix(X_eval_scaled)
            y_pred_raw = model.predict(dtest)
            # Convert binary predictions to probability format
            y_pred_proba = np.column_stack([1 - y_pred_raw, y_pred_raw])

        predictions[i] = y_pred_proba

    return predictions


def load_physionet_data(
    model_type="vanilla", n_samples=1000, n_fractions=10,
    missingness_dir="/home/antonxue/shailesh/MCal/data/tabular/missingness_levels", balanced=True,
    missingness_range="0-30", model_path=None
):
    """
    Load PhysioNet data following MCal benchmark pattern.

    Args:
        model_type (str): "vanilla", "mean_imputation", "zero_imputation", "xgboost_native"
        n_samples (int): Number of samples to use
        n_fractions (int): Number of ablation fractions
        missingness_dir (str): Directory with PhysioNet missingness files
        balanced (bool): Whether to balance the dataset
        missingness_range (str): "0-30", "30-100", or "full"
        model_path (str): Path to trained model (optional)

    Returns:
        tuple: (predictions, labels) where predictions is (n_fractions, n_samples, n_classes)
    """
    print(f"Loading PhysioNet data with {n_samples} samples, {n_fractions} fractions...")
    print(f"Model type: {model_type}, Balanced: {balanced}, Range: {missingness_range}")

    # Load or create balanced dataset
    balanced_file_path = f"balanced_physionet_dataset_{missingness_range}.csv"
    df_balanced = load_or_create_balanced_physionet_dataset(
        missingness_dir=missingness_dir,
        balanced_file_path=balanced_file_path,
        missingness_range=missingness_range
    )

    # Limit samples if requested
    if n_samples < len(df_balanced):
        df_balanced = df_balanced.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Separate features and labels
    X = df_balanced.drop('In-hospital_death', axis=1)
    y = df_balanced['In-hospital_death']

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Generate removal fractions
    removal_fractions = np.linspace(0, 0.9, n_fractions)

    # Load or train model (implement model loading/training logic)
    if model_type == "retrain":
        model = load_or_train_xgboost_model(X, y, model_type, model_path, missingness_dir, missingness_range)
    else:
        model = load_or_train_xgboost_model(X, y, model_type, model_path)

    # Generate fractionwise predictions
    imputation_strategy = {
        "vanilla": "mean",
        "mean_imputation": "mean",
        "zero_imputation": "zero",
        "xgboost_native": "xgboost_native"
    }.get(model_type, "mean")

    predictions = generate_fractionwise_predictions(
        model, X, y, removal_fractions, imputation_strategy
    )

    print(f"Generated predictions shape: {predictions.shape}")
    print(f"Labels shape: {y.values.shape}")

    # Convert to torch tensors
    predictions_tensor = torch.from_numpy(predictions).float()
    labels_tensor = torch.from_numpy(y.values).long()

    return predictions_tensor, labels_tensor


def load_or_train_xgboost_model(X, y, model_type, model_path=None, missingness_dir=None, missingness_range=None):
    """
    Load or train XGBoost model for PhysioNet.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        model_type (str): Type of model
        model_path (str): Path to saved model
        missingness_dir (str): Directory containing missingness level files (for retrain)
        missingness_range (str): Range of missingness levels to use (for retrain)

    Returns:
        Trained XGBoost model
    """
    # This is a placeholder - implement actual model loading/training
    from sklearn.model_selection import train_test_split

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'tree_method': 'hist',
        'enable_categorical': False
    }

    # Train model
    if model_type == "retrain":
        # Train on missing data from missingness_levels files
        from xgboost_utils import train_xgboost_on_missing_data
        predictor = train_xgboost_on_missing_data(
            missingness_dir=missingness_dir,
            imputation_strategy="mean",
            missingness_range=missingness_range
        )
        model = predictor.model
        # Store preprocessing for later use
        model.imputer = predictor.imputer
        model.scaler = predictor.scaler
    elif model_type == "xgboost_native":
        # Use XGBoost's native missing value handling
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
    else:
        # Use imputation
        imputer = SimpleImputer(strategy='mean' if 'mean' in model_type else 'constant')
        if 'zero' in model_type:
            imputer.set_params(fill_value=0)

        X_train_imputed = imputer.fit_transform(X_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train_scaled, y_train)

        # Store preprocessing for later use
        model.imputer = imputer
        model.scaler = scaler

    print(f"Trained XGBoost model for {model_type}")
    return model


def test_physionet_data_loading():
    """Test PhysioNet data loading functionality."""
    print("Testing PhysioNet data loading...")

    try:
        # Test basic data loading
        predictions, labels = load_physionet_data(
            model_type="vanilla",
            n_samples=100,  # Small sample for testing
            n_fractions=5,
            balanced=True
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
    # Test the data loading functionality
    test_physionet_data_loading()