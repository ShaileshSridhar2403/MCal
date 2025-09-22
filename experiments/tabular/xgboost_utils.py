#!/usr/bin/env python3
"""
XGBoost Utilities for MCal Tabular Benchmarks

This module provides XGBoost model training, evaluation, and prediction utilities
specifically designed for missing data robustness and MCal integration.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import pickle
from pathlib import Path


class MCALXGBoostPredictor:
    """
    XGBoost predictor compatible with MCal framework.
    Handles missing data through various imputation strategies.
    """

    def __init__(self, imputation_strategy="mean", xgb_params=None):
        """
        Initialize XGBoost predictor.

        Args:
            imputation_strategy (str): "mean", "zero", or "xgboost_native"
            xgb_params (dict): XGBoost parameters
        """
        self.imputation_strategy = imputation_strategy
        self.model = None
        self.imputer = None
        self.scaler = None

        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
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

    def fit(self, X, y, validation_split=0.2, n_epochs=10, removal_fractions=None):
        """
        Train XGBoost model with missingness robustness.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            validation_split (float): Fraction for validation
            n_epochs (int): Training epochs with missingness
            removal_fractions (list): Fractions for missingness training
        """
        if removal_fractions is None:
            removal_fractions = np.linspace(0, 0.8, 9)  # 0% to 80% missingness

        print(f"Training XGBoost with {self.imputation_strategy} imputation strategy")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Initialize preprocessing
        if self.imputation_strategy == "mean":
            self.imputer = SimpleImputer(strategy='mean')
            X_train_processed = self.imputer.fit_transform(X_train)
        elif self.imputation_strategy == "zero":
            self.imputer = SimpleImputer(strategy='constant', fill_value=0)
            X_train_processed = self.imputer.fit_transform(X_train)
        elif self.imputation_strategy == "xgboost_native":
            self.imputer = None
            X_train_processed = X_train.values
        else:
            raise ValueError(f"Invalid imputation strategy: {self.imputation_strategy}")

        # Fit scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_processed)

        # Initialize model
        self.model = xgb.XGBClassifier(**self.xgb_params)

        # Train with missingness robustness
        for epoch in range(n_epochs):
            print(f"Training epoch {epoch + 1}/{n_epochs}")

            for fraction in removal_fractions:
                # Create missing data
                if fraction > 0:
                    X_missing = self._randomly_remove_features(X_train, fraction)
                else:
                    X_missing = X_train.copy()

                # Apply preprocessing
                if self.imputation_strategy in ["mean", "zero"]:
                    X_missing_processed = self.imputer.transform(X_missing)
                    X_missing_scaled = self.scaler.transform(X_missing_processed)
                elif self.imputation_strategy == "xgboost_native":
                    X_missing_scaled = self.scaler.transform(X_missing)

                # Train incrementally
                if epoch == 0 and fraction == removal_fractions[0]:
                    # First training
                    self.model.fit(X_missing_scaled, y_train)
                else:
                    # Incremental training
                    self.model.fit(
                        X_missing_scaled, y_train,
                        xgb_model=self.model.get_booster()
                    )

        # Final evaluation
        val_accuracy = self.evaluate(X_val, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")

    def predict_proba(self, X):
        """
        Generate probability predictions.

        Args:
            X (pd.DataFrame): Features

        Returns:
            np.ndarray: Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Apply preprocessing
        if self.imputation_strategy in ["mean", "zero"]:
            X_processed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_processed)
        elif self.imputation_strategy == "xgboost_native":
            X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)

    def predict_fractionwise(self, X, removal_fractions=None):
        """
        Generate fractionwise predictions for MCal.

        Args:
            X (pd.DataFrame): Features
            removal_fractions (list): Fractions of features to remove

        Returns:
            np.ndarray: Predictions of shape (n_fractions, n_samples, n_classes)
        """
        if removal_fractions is None:
            removal_fractions = np.linspace(0, 0.9, 10)

        n_samples = len(X)
        n_classes = 2  # Binary classification
        n_fractions = len(removal_fractions)

        predictions = np.zeros((n_fractions, n_samples, n_classes))

        for i, fraction in enumerate(tqdm(removal_fractions, desc="Generating fractionwise predictions")):
            # Apply missing data
            if fraction == 0:
                X_eval = X.copy()
            else:
                X_eval = self._randomly_remove_features(X, fraction)

            # Generate predictions
            y_pred_proba = self.predict_proba(X_eval)
            predictions[i] = y_pred_proba

        return predictions

    def evaluate(self, X, y):
        """
        Evaluate model performance.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True labels

        Returns:
            float: Accuracy score
        """
        y_pred = self.model.predict(self.scaler.transform(
            self.imputer.transform(X) if self.imputer else X
        ))
        return accuracy_score(y, y_pred)

    def save(self, model_path):
        """Save model and preprocessing components."""
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(model_path))

        # Save preprocessing components
        preprocessing_path = model_path.replace('.json', '_preprocessing.pkl')
        with open(preprocessing_path, 'wb') as f:
            pickle.dump({
                'imputer': self.imputer,
                'scaler': self.scaler,
                'imputation_strategy': self.imputation_strategy,
                'xgb_params': self.xgb_params
            }, f)

        print(f"Model saved to {model_path}")
        print(f"Preprocessing saved to {preprocessing_path}")

    def load(self, model_path):
        """Load model and preprocessing components."""
        # Load XGBoost model
        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.model.load_model(model_path)

        # Load preprocessing components
        preprocessing_path = model_path.replace('.json', '_preprocessing.pkl')
        with open(preprocessing_path, 'rb') as f:
            preprocessing = pickle.load(f)

        self.imputer = preprocessing['imputer']
        self.scaler = preprocessing['scaler']
        self.imputation_strategy = preprocessing['imputation_strategy']

        print(f"Model loaded from {model_path}")

    def _randomly_remove_features(self, X, fraction):
        """Randomly remove features to simulate missingness."""
        X_removed = X.copy()
        for column in X_removed.columns:
            mask = np.random.random(len(X_removed)) < fraction
            X_removed.loc[mask, column] = np.nan
        return X_removed


def train_xgboost_with_missingness(X, y, imputation_strategy="mean",
                                 n_epochs=10, removal_fractions=None,
                                 model_save_path=None):
    """
    Train XGBoost model with missingness robustness.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        imputation_strategy (str): Imputation strategy
        n_epochs (int): Training epochs
        removal_fractions (list): Missingness fractions for training
        model_save_path (str): Path to save model

    Returns:
        MCALXGBoostPredictor: Trained model
    """
    if removal_fractions is None:
        removal_fractions = np.linspace(0, 0.8, 9)

    # Initialize predictor
    predictor = MCALXGBoostPredictor(imputation_strategy=imputation_strategy)

    # Train model
    predictor.fit(X, y, n_epochs=n_epochs, removal_fractions=removal_fractions)

    # Save model if path provided
    if model_save_path:
        predictor.save(model_save_path)

    return predictor


def load_or_train_xgboost_model(X, y, model_type="vanilla", model_path=None,
                               model_dir="dataset_store/tabular/models",
                               overwrite=False):
    """
    Load existing XGBoost model or train a new one.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        model_type (str): Type of model ("vanilla", "mean_imputation", etc.)
        model_path (str): Specific model path
        model_dir (str): Directory for model storage
        overwrite (bool): Whether to retrain existing model

    Returns:
        MCALXGBoostPredictor: Trained model
    """
    # Determine imputation strategy from model type
    imputation_strategy_map = {
        "vanilla": "mean",
        "mean_imputation": "mean",
        "zero_imputation": "zero",
        "xgboost_native": "xgboost_native",
        "retrain": "mean"
    }
    imputation_strategy = imputation_strategy_map.get(model_type, "mean")

    # Determine model path
    if model_path is None:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"xgboost_{model_type}.json")

    # Check if model exists
    if not overwrite and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        predictor = MCALXGBoostPredictor(imputation_strategy=imputation_strategy)
        predictor.load(model_path)
        return predictor

    # Train new model
    print(f"Training new XGBoost model ({model_type})")

    if model_type == "retrain":
        # For retrain, we need missingness_dir parameter
        # This will be handled in physionet_data_setup.py
        predictor = train_xgboost_with_missingness(
            X, y,
            imputation_strategy=imputation_strategy,
            model_save_path=model_path
        )
    else:
        predictor = train_xgboost_with_missingness(
            X, y,
            imputation_strategy=imputation_strategy,
            model_save_path=model_path
        )

    return predictor


def train_xgboost_on_missing_data(missingness_dir, imputation_strategy="mean",
                                 model_save_path=None, missingness_range="0-30"):
    """
    Train XGBoost model on binomially sampled missing data from missingness_levels files.

    Args:
        missingness_dir (str): Directory containing missingness level files
        imputation_strategy (str): Imputation strategy for training
        model_save_path (str): Path to save trained model
        missingness_range (str): Range of missingness levels to use for training

    Returns:
        MCALXGBoostPredictor: Trained model
    """
    from physionet_data_setup import load_and_create_missingness_train, load_and_create_missingness_full

    print(f"Training XGBoost on missing data (range: {missingness_range})")

    # Load training data with missing patterns
    if missingness_range == "0-30":
        df = load_and_create_missingness_train(missingness_dir)
    elif missingness_range == "full":
        df = load_and_create_missingness_full(missingness_dir)
    else:
        # Load all files in range
        df_list = []
        start_pct, end_pct = map(int, missingness_range.split('-'))
        for i in range(start_pct, min(end_pct + 10, 100), 10):
            file_pattern = f"missingness_{i:03d}_{i+10:03d}.csv.gz"
            file_path = Path(missingness_dir) / file_pattern
            if file_path.exists():
                df_part = pd.read_csv(file_path)
                df_list.append(df_part)
                print(f"Loaded {len(df_part)} samples from {file_pattern}")

        if df_list:
            df = pd.concat(df_list, ignore_index=True)
        else:
            raise ValueError(f"No missingness files found for range {missingness_range}")

    if df is None or len(df) == 0:
        raise ValueError(f"No training data available for missingness range {missingness_range}")

    print(f"Training on {len(df)} samples with natural missing patterns")

    # Prepare features and labels (consistent with physionet_data_setup.py)
    y_column = 'In-hospital_death'

    # Remove only the target column from features (consistent with standard approach)
    X = df.drop(y_column, axis=1)
    y = df[y_column]

    # Initialize predictor
    predictor = MCALXGBoostPredictor(imputation_strategy=imputation_strategy)

    # Train on missing data (no artificial removal since data already has natural patterns)
    predictor.fit(X, y, validation_split=0.2, n_epochs=1, removal_fractions=[0.0])

    # Save model if path provided
    if model_save_path:
        predictor.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    return predictor


def evaluate_model_robustness(model, X_test, y_test, removal_fractions=None):
    """
    Evaluate model robustness to missing data.

    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        removal_fractions (list): Missingness fractions to test

    Returns:
        dict: Robustness evaluation results
    """
    if removal_fractions is None:
        removal_fractions = np.linspace(0, 0.9, 10)

    results = {
        'removal_fractions': removal_fractions,
        'accuracies': [],
        'auc_scores': []
    }

    for fraction in tqdm(removal_fractions, desc="Evaluating robustness"):
        # Apply missing data
        if fraction == 0:
            X_eval = X_test.copy()
        else:
            X_eval = model._randomly_remove_features(X_test, fraction)

        # Generate predictions
        y_pred_proba = model.predict_proba(X_eval)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        results['accuracies'].append(accuracy)
        results['auc_scores'].append(auc)

        print(f"Fraction {fraction:.1f}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")

    return results


def save_model_outputs(model, X, y, output_path, removal_fractions=None):
    """
    Save model outputs for different removal fractions.

    Args:
        model: Trained model
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        output_path (str): Path to save outputs
        removal_fractions (list): Removal fractions

    Returns:
        str: Path to saved outputs
    """
    # Generate fractionwise predictions
    predictions = model.predict_fractionwise(X, removal_fractions)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save predictions
    np.save(output_path, predictions)
    print(f"Model outputs saved to {output_path}")

    return output_path


if __name__ == "__main__":
    # Test XGBoost utilities
    print("Testing XGBoost utilities...")

    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randint(0, 2, n_samples))

    try:
        # Test model training
        predictor = train_xgboost_with_missingness(X, y, n_epochs=2)
        print("✓ Model training successful")

        # Test fractionwise predictions
        predictions = predictor.predict_fractionwise(X.iloc[:100])
        print(f"✓ Fractionwise predictions shape: {predictions.shape}")

        # Test robustness evaluation
        robustness = evaluate_model_robustness(predictor, X.iloc[:100], y.iloc[:100])
        print(f"✓ Robustness evaluation completed")

        print("✓ All XGBoost utility tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()