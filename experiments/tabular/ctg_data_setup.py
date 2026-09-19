#!/usr/bin/env python3
"""
Simplified CTG Data Setup for MCal Tabular Benchmarks

This module provides a clean, simplified data loading interface for CTG
following the PhysioNet data setup pattern. Focuses on fractionwise MCAR
missing data ablation with CTG-specific preprocessing.
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from tqdm import tqdm
import xgboost as xgb
from mcal.paths import DATA_ROOT

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent


def preprocess_ctg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess CTG features for analysis.
    (Preserved from original CTG implementation)

    Args:
        df (pd.DataFrame): Raw CTG dataset

    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    df_processed = df.copy()

    # Remove metadata columns
    metadata_cols = ['FileName', 'Date', 'SegFile', 'b', 'e']
    df_processed = df_processed.drop(columns=[col for col in metadata_cols if col in df_processed.columns])

    # Remove CLASS column (we use NSP for 3-class classification)
    if 'CLASS' in df_processed.columns:
        df_processed = df_processed.drop(columns=['CLASS'])

    # Handle remaining missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())

    print(f"Preprocessed CTG features: {df_processed.shape[1]-1} features + target")
    print(f"Features: {[col for col in df_processed.columns if col != 'NSP']}")

    return df_processed


def balance_ctg_dataset(df, target_column='NSP'):
    """
    Balance CTG dataset by undersampling majority classes.
    (Adapted from PhysioNet's balance_physionet_dataset for 3-class)
    """
    # Get class counts
    class_counts = df[target_column].value_counts().sort_index()
    print(f"Original class distribution:")
    for class_val, count in class_counts.items():
        print(f"  Class {class_val}: {count}")

    # Undersample all classes to match minority class
    min_count = class_counts.min()

    balanced_dfs = []
    for class_val in sorted(df[target_column].unique()):
        class_df = df[df[target_column] == class_val]
        if len(class_df) > min_count:
            class_df = resample(class_df, replace=False, n_samples=min_count, random_state=42)
        balanced_dfs.append(class_df)

    # Combine and shuffle
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    balanced_counts = df_balanced[target_column].value_counts().sort_index()
    print(f"Balanced class distribution:")
    for class_val, count in balanced_counts.items():
        print(f"  Class {class_val}: {count}")

    return df_balanced


def load_clean_ctg_data(data_path, n_samples=1000):
    """Load and balance CTG dataset using existing preprocessing"""
    # Load CTG dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CTG dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded CTG dataset: {df.shape[0]} samples, {df.shape[1]} features")

    # Remove rows with missing target values
    df = df.dropna(subset=['NSP']).copy()

    # Apply CTG-specific preprocessing
    df_processed = preprocess_ctg_features(df)

    # Convert NSP labels from (1,2,3) to (0,1,2) for standard classification
    df_processed['NSP'] = df_processed['NSP'] - 1
    print(f"Converted NSP labels: {sorted(df_processed['NSP'].unique())} (0=Normal, 1=Suspect, 2=Pathological)")

    # Apply simple balancing (undersample majority classes)
    df_balanced = balance_ctg_dataset(df_processed)

    # Sample if needed
    if n_samples < len(df_balanced):
        df_balanced = df_balanced.sample(n=n_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {n_samples} samples from balanced dataset")

    return df_balanced


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
        'objective': 'multi:softprob',  # Multi-class probabilities
        'num_class': 3,                 # 3 classes for CTG
        'eval_metric': 'mlogloss',      # Multi-class log loss
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
        - Input: 100 samples x 21 features (all complete)
        - removal_fraction = 0.3
        - Output: 100 samples x 21 features (30% randomly set to NaN per sample)
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


def load_ctg_data(model_type="vanilla", fill_value="mean", n_samples=1000, n_fractions=10,
                  data_path=str(DATA_ROOT / "tabular" / "ctg" / "ctg_dataset.csv"),
                  missing_value=None):
    """
    Simple, clean CTG data loading following PhysioNet pattern.

    Args:
        model_type: "vanilla" or "retrained" - controls training data selection
        fill_value: "mean", "nan", "zero", or "-10" - how to handle missing values during prediction
        n_samples: Number of samples to use
        n_fractions: Number of ablation fractions
        data_path: Path to CTG dataset CSV file
        missing_value: Value to treat as missing in XGBoost (e.g., -10, np.nan)

    Returns:
        (predictions, labels): Torch tensors with shapes (k, n, c) and (n,)
        where c=3 for CTG (3-class classification)
    """
    print(f"Loading CTG data with {n_samples} samples, {n_fractions} fractions...")
    print(f"Model type: {model_type}, Fill value: {fill_value}")

    # 1. Load clean base data
    clean_data = load_clean_ctg_data(data_path, n_samples)
    X_clean = clean_data.drop('NSP', axis=1)
    y_clean = clean_data['NSP']

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


def test_ctg_loading():
    """Test CTG data loading functionality."""
    print("Testing simplified CTG data loading...")

    try:
        # Test basic data loading
        predictions, labels = load_ctg_data(
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
        assert predictions.shape[2] == 3, "Should be 3-class classification"

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the simplified data loading functionality
    test_ctg_loading()