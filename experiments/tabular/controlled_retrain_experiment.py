#!/usr/bin/env python3
"""
Controlled Retrain Experiment for PhysioNet Data

This script implements a controlled comparison between MCal_CE and retrain methods
by using the same base model and controlled missing data generation.

Experiment Design:
1. Extract 0-30% missingness data from missingness_levels/
2. Train base XGBoost model on this data
3. Generate controlled MCAR missing data for evaluation
4. Compare MCal_CE (calibration) vs retrain (retraining) approaches
"""

import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add MCal imports
import sys
from mcal.calibrators.mcal_ce import MCal_CE
from mcal.utils.optimization import kl_divergence, get_expectation


class ControlledRetrainExperiment:
    """
    Controlled experiment comparing MCal_CE vs retrain on PhysioNet data.
    """

    def __init__(self, missingness_dir="/home/antonxue/shailesh/MCal/data/tabular/missingness_levels",
                 device="cuda", random_state=42):
        """
        Initialize the experiment.

        Args:
            missingness_dir (str): Directory containing missingness level files
            device (str): Device for computation
            random_state (int): Random seed
        """
        self.missingness_dir = Path(missingness_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.base_model = None
        self.base_imputer = None
        self.base_scaler = None
        self.base_data = None

        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        print(f"Initialized ControlledRetrainExperiment")
        print(f"Device: {self.device}")
        print(f"Missingness dir: {self.missingness_dir}")

    def extract_base_dataset(self):
        """
        Extract and combine 0-30% missingness data to create base dataset.

        Returns:
            pd.DataFrame: Combined base dataset
        """
        print("\\n=== Phase 1: Extracting Base Dataset (0-30% missingness) ===")

        base_data_list = []
        total_samples = 0

        # Extract data from 0-30% missingness ranges
        for i in [0, 10, 20]:  # 0-10%, 10-20%, 20-30%
            filename = f'missingness_{i:03d}_{i+10:03d}.csv.gz'
            filepath = self.missingness_dir / filename

            if filepath.exists():
                df = pd.read_csv(filepath, compression='gzip')
                samples = len(df)
                total_samples += samples
                base_data_list.append(df)
                print(f"  Loaded {filename}: {samples} samples, {len(df.columns)} features")
            else:
                print(f"  Warning: {filename} not found")

        if not base_data_list:
            raise ValueError("No base data files found!")

        # Combine all base data
        self.base_data = pd.concat(base_data_list, ignore_index=True)
        print(f"  Combined base dataset: {len(self.base_data)} samples, {len(self.base_data.columns)} features")

        # Show data distribution
        print(f"  Total samples by range:")
        current_pos = 0
        for i, df in enumerate(base_data_list):
            range_name = f"{i*10}-{(i+1)*10}%"
            print(f"    {range_name}: {len(df)} samples ({len(df)/total_samples*100:.1f}%)")

        return self.base_data

    def prepare_features_labels(self, df):
        """
        Prepare features and labels from dataframe.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            tuple: (X, y) features and labels
        """
        # Look for common target column names
        target_cols = ['target', 'label', 'y', 'outcome', 'class', 'mortality', 'died', 'death']
        target_col = None

        for col in target_cols:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            # For PhysioNet, check for mortality-related columns
            mortality_cols = [col for col in df.columns if 'mort' in col.lower() or 'death' in col.lower() or 'died' in col.lower()]
            if mortality_cols:
                target_col = mortality_cols[0]
                print(f"  Found mortality column '{target_col}' as target")
            else:
                # Use a reasonable continuous target and binarize it
                target_col = df.columns[-1]
                print(f"  Using last column '{target_col}' as target (will binarize)")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle missing labels
        print(f"  Original labels - Missing: {y.isnull().sum()}, Total: {len(y)}")
        if y.isnull().any():
            print(f"  Dropping {y.isnull().sum()} samples with missing labels")
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]

        # Convert to binary classification if needed
        if len(y.unique()) > 2:
            print(f"  Converting to binary classification (threshold = median)")
            threshold = y.median()
            y_binary = (y > threshold).astype(int)
            print(f"  Threshold: {threshold}")
            print(f"  Class 0 (≤{threshold}): {(y_binary == 0).sum()} samples")
            print(f"  Class 1 (>{threshold}): {(y_binary == 1).sum()} samples")
            y = y_binary

        print(f"  Final features shape: {X.shape}")
        print(f"  Final labels shape: {y.shape}")
        print(f"  Final label distribution: {y.value_counts().to_dict()}")

        return X, y

    def train_base_model(self):
        """
        Train base XGBoost model on the extracted 0-30% missingness data.

        Returns:
            dict: Training results and model info
        """
        print("\\n=== Phase 1: Training Base Model ===")

        if self.base_data is None:
            raise ValueError("Base data not extracted. Call extract_base_dataset() first.")

        # Prepare features and labels
        X, y = self.prepare_features_labels(self.base_data)

        # Handle missing values in base data (it already has some missingness)
        print("  Handling missing values in base data...")
        self.base_imputer = SimpleImputer(strategy='mean')
        X_imputed = self.base_imputer.fit_transform(X)

        # Scale features
        print("  Scaling features...")
        self.base_scaler = StandardScaler()
        X_scaled = self.base_scaler.fit_transform(X_imputed)

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # XGBoost parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': self.random_state,
            'tree_method': 'hist',
            'enable_categorical': False
        }

        # Train base model
        print("  Training XGBoost base model...")
        self.base_model = xgb.XGBClassifier(**xgb_params)
        self.base_model.fit(X_train, y_train)

        # Evaluate base model
        train_score = self.base_model.score(X_train, y_train)
        val_score = self.base_model.score(X_val, y_val)

        # Get predictions for AUC
        y_pred_proba = self.base_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_pred_proba)

        results = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'val_auc': val_auc,
            'n_features': X.shape[1],
            'n_samples': len(X),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }

        print(f"  Base model training completed:")
        print(f"    Train accuracy: {train_score:.4f}")
        print(f"    Val accuracy: {val_score:.4f}")
        print(f"    Val AUC: {val_auc:.4f}")
        print(f"    Features: {X.shape[1]}")
        print(f"    Training samples: {len(X_train)}")

        return results

    def save_base_model(self, save_path="controlled_experiment_base_model.pkl"):
        """
        Save the trained base model and preprocessing components.

        Args:
            save_path (str): Path to save the model
        """
        if self.base_model is None:
            raise ValueError("Base model not trained yet.")

        model_data = {
            'model': self.base_model,
            'imputer': self.base_imputer,
            'scaler': self.base_scaler,
            'random_state': self.random_state,
            'feature_names': self.base_data.columns[:-1].tolist()  # Exclude target
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"  Base model and preprocessing saved to: {save_path}")

        return save_path


def main():
    """
    Main function to run Phase 1: Extract data and train base model.
    """
    print("="*60)
    print("Controlled Retrain Experiment - Phase 1")
    print("="*60)

    # Initialize experiment
    experiment = ControlledRetrainExperiment()

    try:
        # Phase 1a: Extract base dataset
        base_data = experiment.extract_base_dataset()

        # Phase 1b: Train base model
        results = experiment.train_base_model()

        # Phase 1c: Save base model
        model_path = experiment.save_base_model()

        print("\\n" + "="*60)
        print("Phase 1 COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Base dataset: {len(base_data)} samples")
        print(f"Base model trained with validation AUC: {results['val_auc']:.4f}")
        print(f"Model saved to: {model_path}")
        print("\\nReady for Phase 2: Controlled MCAR evaluation")

        return True

    except Exception as e:
        print(f"\\nERROR in Phase 1: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)