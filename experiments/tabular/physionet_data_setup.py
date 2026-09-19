#!/usr/bin/env python3
"""
Simplified PhysioNet Data Setup for MCal Tabular Benchmarks

This module provides a clean, simplified data loading interface for PhysioNet
following the MRI data setup pattern. Reduced from 816 lines to ~200 lines.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from tqdm import tqdm
import xgboost as xgb

# Add MCal to path
mcal_root = Path(__file__).parent.parent.parent


def load_and_create_missingness_train(missingness_dir):
    """
    Load data from missingness level files and create a training dataframe
    with data falling in the 0-30% missingness range.
    (Kept from original - used as base data source)
    """
    # List all relevant CSV files
    file_pattern = os.path.join(missingness_dir, "missingness_0*.csv.gz")
    files = glob.glob(file_pattern)

    # Filter files for 0-30% missingness
    files = [f for f in files if int(os.path.basename(f).split('_')[1][:3]) <= 30]

    if not files:
        print("No files found for 0-30% missingness range.")
        return None

    # Load and concatenate dataframes
    dfs = []
    for file in files:
        df = pd.read_csv(file, compression='gzip', index_col=0)
        dfs.append(df)

    train_df = pd.concat(dfs, axis=0)

    print(f"Created training dataframe with shape: {train_df.shape}")
    print(f"Missingness range: 0-30%")

    return train_df


def balance_physionet_dataset(df, target_column='In-hospital_death'):
    """
    Balance PhysioNet dataset by undersampling majority class.
    (Simplified from original)
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


def load_clean_physionet_data(missingness_dir, n_samples=1000):
    """Load and balance PhysioNet dataset using existing train function"""
    # Use load_and_create_missingness_train() for 0-30% missingness base data
    df = load_and_create_missingness_train(missingness_dir)
    if df is None:
        raise ValueError("Failed to load PhysioNet data")

    # Apply simple balancing (undersample majority class)
    df_balanced = balance_physionet_dataset(df)

    # Sample if needed
    if n_samples < len(df_balanced):
        df_balanced = df_balanced.sample(n=n_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {n_samples} samples from balanced dataset")

    return df_balanced


def train_xgboost_model(X_train, y_train, missing_value=None):
    """
    Train XGBoost model with optional custom missing value handling.

    Args:
        X_train: Training features
        y_train: Training labels
        missing_value: Value to treat as missing (e.g., -10, np.nan)

    Returns:
        Trained XGBoost model
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
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

    Args:
        data: DataFrame or array with complete features
        removal_fraction: Float between 0-1, fraction of features to remove per sample

    Returns:
        DataFrame/array with randomly missing values

    Example:
        - Input: 100 samples x 40 features (all complete)
        - removal_fraction = 0.3
        - Output: 100 samples x 40 features (30% randomly set to NaN per sample)
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


def load_physionet_data(model_type="vanilla", fill_value="mean", n_samples=1000, n_fractions=10,
                       missingness_dir=str(Path(__file__).parent.parent.parent / "data" / "tabular" / "missingness_levels"),
                       missing_value=None):
    """
    Simple, clean PhysioNet data loading following MRI pattern.

    Args:
        model_type: "vanilla" or "retrained" - controls training data selection
        fill_value: "mean", "nan", "zero", or "-10" - how to handle missing values during prediction
        n_samples: Number of samples to use
        n_fractions: Number of ablation fractions
        missing_value: Value to treat as missing in XGBoost (e.g., -10, np.nan)

    Returns:
        (predictions, labels): Torch tensors with shapes (k, n, c) and (n,)
    """
    print(f"Loading PhysioNet data with {n_samples} samples, {n_fractions} fractions...")
    print(f"Model type: {model_type}, Fill value: {fill_value}")

    # 1. Load clean base data
    clean_data = load_clean_physionet_data(missingness_dir, n_samples)
    X_clean = clean_data.drop('In-hospital_death', axis=1)
    y_clean = clean_data['In-hospital_death']

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
    imputer = SimpleImputer(strategy='mean')  # default is mean and hence we impute with mean
    X_train = imputer.fit_transform(X_train)

    # 3. Train model
    model = train_xgboost_model(X_train, y_clean, missing_value=missing_value)

    # 4. Generate predictions across ablation fractions
    ablation_fractions = [i/n_fractions for i in range(n_fractions)]
    all_probs = []

    print(f"Generating predictions across {n_fractions} ablation fractions...")

    # Initialize imputer on clean data to ensure consistent feature count
    # if fill_value == "mean":
    #     imputer = SimpleImputer(strategy='mean')
    #     imputer.fit(X_clean)

    for fraction in tqdm(ablation_fractions, desc="Ablation fractions"):
        # Apply missing data simulation
        data_with_missing = apply_missing_data_simulation(X_clean, fraction)

        # Apply preprocessing based on fill_value - INLINE CODE
        if fill_value == "mean":
            # Mean imputation only (no scaling needed for XGBoost)
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


def test_physionet_loading():
    """Test PhysioNet data loading functionality."""
    print("Testing simplified PhysioNet data loading...")

    try:
        # Test basic data loading
        predictions, labels = load_physionet_data(
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
    test_physionet_loading()