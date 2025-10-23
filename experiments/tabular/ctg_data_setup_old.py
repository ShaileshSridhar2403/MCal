#!/usr/bin/env python3
"""
CTG (Cardiotocography) Dataset Setup for MCal Tabular Benchmarks

This module provides CTG dataset loading, preprocessing, and medical-realistic
missing data ablation for 3-class fetal monitoring classification.

Dataset Details:
- Source: UCI Machine Learning Repository
- Samples: 2,129 fetal cardiotocograms (CTGs)
- Features: 21 medical features
- Classes: 3 (Normal=1, Suspect=2, Pathological=3)
- Task: Multi-class classification of fetal heart rate patterns
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_ctg_dataset(data_path: str = "/home/antonxue/shailesh/MCal/data/tabular/ctg/ctg_dataset.csv") -> pd.DataFrame:
    """
    Load CTG dataset from CSV file.

    Args:
        data_path (str): Path to CTG dataset CSV file

    Returns:
        pd.DataFrame: Loaded CTG dataset
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CTG dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Remove rows with missing target values
    df = df.dropna(subset=['NSP']).copy()

    print(f"Loaded CTG dataset: {df.shape[0]} samples, {df.shape[1]} features")
    return df


def get_ctg_feature_groups() -> Dict[str, List[str]]:
    """
    Define CTG feature groups for medical-realistic ablation.

    Returns:
        Dict[str, List[str]]: Feature groups based on medical equipment/measurements
    """
    feature_groups = {
        # Baseline fetal heart rate measurements
        'baseline_fhr': ['LB', 'LBE'],

        # Fetal heart rate variability (time domain)
        'fhr_variability': ['ASTV', 'MSTV', 'ALTV', 'MLTV'],

        # Accelerations and movements
        'accelerations': ['AC', 'FM'],

        # Uterine contractions
        'uterine_contractions': ['UC'],

        # Decelerations (critical for pathology detection)
        'decelerations': ['DL', 'DS', 'DP', 'DR'],

        # Histogram analysis (automated processing features)
        'histogram_features': ['Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency'],

        # Classification features (expert-derived)
        'expert_features': ['A', 'B', 'C', 'D', 'E'],

        # Additional derived features
        'derived_features': ['AD', 'DE', 'LD', 'FS', 'SUSP']
    }

    return feature_groups


def preprocess_ctg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess CTG features for analysis.

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


def balance_ctg_classes(df: pd.DataFrame, strategy: str = "undersample", random_state: int = 42) -> pd.DataFrame:
    """
    Balance CTG classes for training.

    Args:
        df (pd.DataFrame): Input dataset
        strategy (str): Balancing strategy ("undersample", "oversample", "none")
        random_state (int): Random seed

    Returns:
        pd.DataFrame: Balanced dataset
    """
    if strategy == "none":
        return df

    # Get class counts
    class_counts = df['NSP'].value_counts().sort_index()
    print(f"Original class distribution: {class_counts.to_dict()}")

    if strategy == "undersample":
        # Undersample to minority class size
        min_count = class_counts.min()

        balanced_dfs = []
        for class_val in sorted(df['NSP'].unique()):
            class_df = df[df['NSP'] == class_val]
            if len(class_df) > min_count:
                class_df = resample(class_df, n_samples=min_count, random_state=random_state)
            balanced_dfs.append(class_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    elif strategy == "oversample":
        # Oversample to majority class size
        max_count = class_counts.max()

        balanced_dfs = []
        for class_val in sorted(df['NSP'].unique()):
            class_df = df[df['NSP'] == class_val]
            if len(class_df) < max_count:
                class_df = resample(class_df, n_samples=max_count, random_state=random_state, replace=True)
            balanced_dfs.append(class_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")

    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    new_class_counts = balanced_df['NSP'].value_counts().sort_index()
    print(f"Balanced class distribution: {new_class_counts.to_dict()}")

    return balanced_df


def split_ctg_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split CTG data into train and test sets with stratification.

    Args:
        df (pd.DataFrame): Input dataset
        test_size (float): Proportion for test set
        random_state (int): Random seed

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['NSP'],
        random_state=random_state
    )

    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")

    return train_df, test_df


# ==================================================
# MEDICAL-REALISTIC MISSING DATA ABLATION FUNCTIONS
# ==================================================

def simulate_equipment_failure(df: pd.DataFrame, failure_rates: Dict[str, float], random_state: int = 42) -> pd.DataFrame:
    """
    Simulate CTG equipment sensor failures with medical realism.

    Args:
        df (pd.DataFrame): Input dataset
        failure_rates (Dict[str, float]): Failure rates for each equipment type
        random_state (int): Random seed

    Returns:
        pd.DataFrame: Dataset with simulated equipment failures
    """
    df_ablated = df.copy()
    np.random.seed(random_state)
    feature_groups = get_ctg_feature_groups()

    for equipment, rate in failure_rates.items():
        if equipment in feature_groups:
            features = feature_groups[equipment]
            # Apply failure to entire feature group (realistic equipment failure)
            mask = np.random.random(len(df_ablated)) < rate
            df_ablated.loc[mask, features] = np.nan

            if rate > 0:
                affected_samples = mask.sum()
                print(f"Equipment failure simulation - {equipment}: {affected_samples} samples affected ({affected_samples/len(df)*100:.1f}%)")

    return df_ablated


def simulate_monitoring_interruptions(df: pd.DataFrame, interruption_patterns: Dict[str, float], random_state: int = 42) -> pd.DataFrame:
    """
    Simulate monitoring interruptions and signal quality issues.

    Args:
        df (pd.DataFrame): Input dataset
        interruption_patterns (Dict[str, float]): Interruption patterns and rates
        random_state (int): Random seed

    Returns:
        pd.DataFrame: Dataset with simulated interruptions
    """
    df_ablated = df.copy()
    np.random.seed(random_state)
    feature_groups = get_ctg_feature_groups()

    # Patient movement - affects variability measurements
    if 'patient_movement' in interruption_patterns:
        rate = interruption_patterns['patient_movement']
        affected_features = feature_groups['fhr_variability']
        mask = np.random.random(len(df_ablated)) < rate
        df_ablated.loc[mask, affected_features] = np.nan
        print(f"Patient movement simulation: {mask.sum()} samples affected")

    # Electrode displacement - affects baseline and accelerations
    if 'electrode_displacement' in interruption_patterns:
        rate = interruption_patterns['electrode_displacement']
        affected_features = feature_groups['baseline_fhr'] + feature_groups['accelerations']
        mask = np.random.random(len(df_ablated)) < rate
        df_ablated.loc[mask, affected_features] = np.nan
        print(f"Electrode displacement simulation: {mask.sum()} samples affected")

    # Processing software issues - affects histogram features
    if 'processing_issues' in interruption_patterns:
        rate = interruption_patterns['processing_issues']
        affected_features = feature_groups['histogram_features']
        mask = np.random.random(len(df_ablated)) < rate
        df_ablated.loc[mask, affected_features] = np.nan
        print(f"Processing issues simulation: {mask.sum()} samples affected")

    return df_ablated


def simulate_protocol_variations(df: pd.DataFrame, protocol_type: str, random_state: int = 42) -> pd.DataFrame:
    """
    Simulate different clinical monitoring protocols.

    Args:
        df (pd.DataFrame): Input dataset
        protocol_type (str): Type of protocol ("basic", "extended", "emergency")
        random_state (int): Random seed

    Returns:
        pd.DataFrame: Dataset with protocol-based feature availability
    """
    df_ablated = df.copy()
    np.random.seed(random_state)
    feature_groups = get_ctg_feature_groups()

    if protocol_type == "basic":
        # Basic monitoring: only core features available
        keep_features = (feature_groups['baseline_fhr'] +
                        feature_groups['accelerations'] +
                        feature_groups['uterine_contractions'] +
                        ['NSP'])  # Always keep target

        all_features = df_ablated.columns.tolist()
        remove_features = [f for f in all_features if f not in keep_features]
        df_ablated[remove_features] = np.nan

        print(f"Basic protocol: keeping {len(keep_features)-1} core features")

    elif protocol_type == "emergency":
        # Emergency protocol: critical features only
        keep_features = (feature_groups['baseline_fhr'] +
                        feature_groups['decelerations'] +
                        feature_groups['uterine_contractions'] +
                        ['NSP'])  # Always keep target

        all_features = df_ablated.columns.tolist()
        remove_features = [f for f in all_features if f not in keep_features]
        df_ablated[remove_features] = np.nan

        print(f"Emergency protocol: keeping {len(keep_features)-1} critical features")

    elif protocol_type == "extended":
        # Extended protocol: all features available (no ablation)
        print("Extended protocol: all features available")

    else:
        raise ValueError(f"Unknown protocol type: {protocol_type}")

    return df_ablated


def apply_missing_ablation(df: pd.DataFrame, strategy: str = "medical", fraction: float = 0.3, random_state: int = 42) -> pd.DataFrame:
    """
    Apply unified missing data ablation with different strategies.

    Args:
        df (pd.DataFrame): Input dataset
        strategy (str): Ablation strategy ("random", "equipment", "temporal", "protocol", "medical")
        fraction (float): Target fraction of missing data
        random_state (int): Random seed

    Returns:
        pd.DataFrame: Dataset with applied ablation
    """
    if strategy == "random":
        # Random feature ablation (baseline)
        df_ablated = df.copy()
        np.random.seed(random_state)

        feature_cols = [col for col in df.columns if col != 'NSP']
        n_features_to_remove = int(len(feature_cols) * fraction)

        for _, row_idx in enumerate(df_ablated.index):
            features_to_remove = np.random.choice(feature_cols, n_features_to_remove, replace=False)
            df_ablated.loc[row_idx, features_to_remove] = np.nan

        print(f"Random ablation: {fraction*100:.1f}% of features removed per sample")

    elif strategy == "equipment":
        # Equipment-based failures
        failure_rates = {
            'fhr_variability': fraction * 0.4,
            'histogram_features': fraction * 0.3,
            'accelerations': fraction * 0.2,
            'derived_features': fraction * 0.1
        }
        df_ablated = simulate_equipment_failure(df, failure_rates, random_state)

    elif strategy == "temporal":
        # Temporal interruptions
        interruption_patterns = {
            'patient_movement': fraction * 0.4,
            'electrode_displacement': fraction * 0.3,
            'processing_issues': fraction * 0.3
        }
        df_ablated = simulate_monitoring_interruptions(df, interruption_patterns, random_state)

    elif strategy == "protocol":
        # Protocol-based variations
        if fraction < 0.3:
            protocol_type = "extended"
        elif fraction < 0.6:
            protocol_type = "basic"
        else:
            protocol_type = "emergency"
        df_ablated = simulate_protocol_variations(df, protocol_type, random_state)

    elif strategy == "medical":
        # Combined medical-realistic approach
        df_ablated = df.copy()

        # Apply multiple realistic patterns
        if fraction > 0.1:
            equipment_rates = {
                'fhr_variability': fraction * 0.25,
                'histogram_features': fraction * 0.2,
                'derived_features': fraction * 0.15
            }
            df_ablated = simulate_equipment_failure(df_ablated, equipment_rates, random_state)

        if fraction > 0.2:
            interruption_patterns = {
                'patient_movement': fraction * 0.2,
                'electrode_displacement': fraction * 0.15
            }
            df_ablated = simulate_monitoring_interruptions(df_ablated, interruption_patterns, random_state + 1)

        print(f"Medical-realistic ablation: {fraction*100:.1f}% target missing rate")

    else:
        raise ValueError(f"Unknown ablation strategy: {strategy}")

    # Calculate actual missing rate
    feature_cols = [col for col in df_ablated.columns if col != 'NSP']
    missing_rate = df_ablated[feature_cols].isnull().sum().sum() / (len(df_ablated) * len(feature_cols))
    print(f"Actual missing rate achieved: {missing_rate*100:.1f}%")

    return df_ablated


# ==================================================
# CTG DATA GENERATION FOR BENCHMARK
# ==================================================

def train_ctg_xgboost_model(X: pd.DataFrame, y: pd.Series, model_type: str = "vanilla") -> 'XGBClassifier':
    """
    Train XGBoost model for CTG multi-class classification.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels (NSP values)
        model_type (str): Model type

    Returns:
        XGBClassifier: Trained model
    """
    try:
        import xgboost as xgb
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("XGBoost is required. Install with: pip install xgboost")

    # Handle missing values
    if X.isnull().any().any():
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_imputed = X.copy()
        imputer = None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train XGBoost for multi-class classification
    xgb_params = {
        'objective': 'multi:softprob',  # Multi-class probabilities
        'num_class': 3,                 # 3 classes (1, 2, 3)
        'eval_metric': 'mlogloss',      # Multi-class log loss
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'tree_method': 'hist',
        # 'missing' :np.nan
    }

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_scaled, y)

    # Store preprocessing for prediction
    model.imputer = imputer
    model.scaler = scaler

    print(f"Trained CTG XGBoost model ({model_type}) - Multi-class accuracy: {model.score(X_scaled, y):.3f}")

    return model


def generate_fractionwise_ctg_predictions(model, X: pd.DataFrame, y: pd.Series,
                                        removal_fractions: np.ndarray,
                                        ablation_strategy: str = "medical") -> np.ndarray:
    """
    Generate CTG predictions across different missing data fractions.

    Args:
        model: Trained XGBoost model
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        removal_fractions (np.ndarray): Fractions of missing data to simulate
        ablation_strategy (str): Strategy for creating missing data

    Returns:
        np.ndarray: Predictions of shape (n_fractions, n_samples, n_classes)
    """
    n_fractions = len(removal_fractions)
    n_samples = len(X)
    n_classes = 3  # CTG has 3 classes

    predictions = np.zeros((n_fractions, n_samples, n_classes))

    print(f"Generating CTG predictions across {n_fractions} missing data fractions...")

    for i, fraction in enumerate(tqdm(removal_fractions, desc="Processing fractions")):
        # Apply missing data ablation
        X_ablated = apply_missing_ablation(X, strategy=ablation_strategy, fraction=fraction, random_state=42+i)

        # Remove target column if present
        if 'NSP' in X_ablated.columns:
            X_ablated = X_ablated.drop(columns=['NSP'])

        # Handle missing values for prediction
        if hasattr(model, 'imputer') and model.imputer is not None:
            X_processed = pd.DataFrame(
                model.imputer.transform(X_ablated),
                columns=X_ablated.columns,
                index=X_ablated.index
            )
        else:
            X_processed = X_ablated.fillna(X_ablated.median())

        # Scale features
        if hasattr(model, 'scaler'):
            X_scaled = model.scaler.transform(X_processed)
        else:
            X_scaled = X_processed.values

        # Get multi-class probabilities
        y_pred_proba = model.predict_proba(X_scaled)
        predictions[i] = y_pred_proba

    print(f"Generated predictions shape: {predictions.shape}")
    return predictions


def load_ctg_data(model_type: str = "vanilla", n_samples: int = 1000, n_fractions: int = 10,
                  ablation_strategy: str = "medical", balanced: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Main function to load CTG data and generate multi-class predictions.

    Args:
        model_type (str): Type of model to train
        n_samples (int): Number of samples to use
        n_fractions (int): Number of missing data fractions
        ablation_strategy (str): Strategy for missing data simulation
        balanced (bool): Whether to balance classes

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, labels)
            - predictions: shape (n_fractions, n_samples, 3)
            - labels: shape (n_samples,)
    """
    print(f"Loading CTG data with {n_samples} samples, {n_fractions} fractions...")
    print(f"Model type: {model_type}, Ablation: {ablation_strategy}, Balanced: {balanced}")

    # Load and preprocess dataset
    df = load_ctg_dataset()
    df = preprocess_ctg_features(df)

    # Balance classes if requested
    if balanced:
        df = balance_ctg_classes(df, strategy="undersample")

    # Sample requested number of samples
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Separate features and labels
    X = df.drop('NSP', axis=1)
    y = df['NSP'].astype(int) - 1  # Convert to 0-based indexing (0, 1, 2)

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

    # Train model
    model = train_ctg_xgboost_model(X, y, model_type)

    # Generate predictions across missing data fractions
    removal_fractions = np.linspace(0, 0.9, n_fractions)
    predictions = generate_fractionwise_ctg_predictions(
        model, X, y, removal_fractions, ablation_strategy
    )

    # Convert to torch tensors
    predictions_tensor = torch.from_numpy(predictions).float()
    labels_tensor = torch.from_numpy(y.values).long()

    print(f"Loaded CTG data - Predictions: {predictions_tensor.shape}, Labels: {labels_tensor.shape}")

    return predictions_tensor, labels_tensor


if __name__ == "__main__":
    # Test the data loading pipeline
    print("Testing CTG data setup...")

    try:
        predictions, labels = load_ctg_data(n_samples=100, n_fractions=5, ablation_strategy="medical")
        print(f"✅ CTG data loading successful!")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {torch.bincount(labels)}")

    except Exception as e:
        print(f"❌ CTG data loading failed: {e}")
        import traceback
        traceback.print_exc()