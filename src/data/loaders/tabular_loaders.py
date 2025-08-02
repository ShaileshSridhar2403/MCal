"""Tabular dataset loaders for Wisconsin breast cancer, Physionet, etc."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List
import logging

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class TabularDataset(Dataset):
    """Generic tabular dataset class."""
    
    def __init__(
        self,
        features: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        feature_names: Optional[List[str]] = None
    ):
        """Initialize tabular dataset.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Labels of shape (n_samples,) (optional for inference)
            feature_names: Names of features (optional)
        """
        if isinstance(features, np.ndarray):
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features.float()
        
        if labels is not None:
            if isinstance(labels, np.ndarray):
                self.labels = torch.tensor(labels, dtype=torch.long)
            else:
                self.labels = labels.long()
        else:
            self.labels = None
        
        self.feature_names = feature_names
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {'features': self.features[idx]}
        
        if self.labels is not None:
            item['label'] = self.labels[idx]
        
        return item
    
    @property
    def num_features(self) -> int:
        return self.features.shape[1]


class TabularDataLoader(BaseDataLoader):
    """Base class for tabular dataset loaders."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        normalize: bool = True
    ):
        super().__init__(data_dir, cache_dir, seed)
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
    
    def create_dataset(
        self,
        features: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> TabularDataset:
        """Create a tabular dataset from features and labels."""
        return TabularDataset(
            features=features,
            labels=labels,
            feature_names=self.feature_names
        )
    
    def preprocess_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess features (normalization, etc.).
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Preprocessed feature matrix
        """
        if self.scaler is not None:
            if fit:
                X_processed = self.scaler.fit_transform(X)
            else:
                X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        
        return X_processed
    
    def preprocess_labels(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess labels (encoding, etc.).
        
        Args:
            y: Label array
            fit: Whether to fit the encoder (True for training data)
            
        Returns:
            Preprocessed label array
        """
        if fit:
            y_processed = self.label_encoder.fit_transform(y)
        else:
            y_processed = self.label_encoder.transform(y)
        
        return y_processed


class WisconsinBreastCancerLoader(TabularDataLoader):
    """Wisconsin breast cancer dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        normalize: bool = True
    ):
        super().__init__(data_dir, cache_dir, seed, normalize)
        self.dataset_name = "wisconsin_breast_cancer"
        self.num_classes = 2
        self.class_names = ["benign", "malignant"]
        self.dataset_info = {
            "description": "Breast cancer classification from cell nuclei features",
            "modality": "tabular",
            "task": "binary_classification",
            "num_features": 30
        }
    
    def download_dataset(self, **kwargs) -> None:
        """Download Wisconsin breast cancer dataset from sklearn."""
        try:
            from sklearn.datasets import load_breast_cancer
            
            # Load dataset
            data = load_breast_cancer()
            
            # Create DataFrame
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
            # Save to CSV
            save_path = self.data_dir / "wisconsin_breast_cancer.csv"
            df.to_csv(save_path, index=False)
            
            logger.info(f"Saved Wisconsin breast cancer dataset to {save_path}")
            
        except ImportError:
            logger.error("scikit-learn not found. Install with: pip install scikit-learn")
        except Exception as e:
            logger.error(f"Failed to download Wisconsin breast cancer dataset: {e}")
    
    def setup_dataset(
        self,
        data_file: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup Wisconsin breast cancer dataset.
        
        Args:
            data_file: Path to CSV file (if None, will download)
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (train_dataset, test_dataset, val_dataset)
        """
        if data_file is None:
            data_file = self.data_dir / "wisconsin_breast_cancer.csv"
        
        # Download if file doesn't exist
        if not Path(data_file).exists():
            self.download_dataset(**kwargs)
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Separate features and labels
        X = df.drop('target', axis=1).values
        y = df['target'].values
        self.feature_names = df.drop('target', axis=1).columns.tolist()
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=self.seed, stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        # Preprocess features
        X_train = self.preprocess_features(X_train, fit=True)
        X_test = self.preprocess_features(X_test, fit=False)
        if X_val is not None:
            X_val = self.preprocess_features(X_val, fit=False)
        
        # Create datasets
        train_dataset = self.create_dataset(X_train, y_train)
        test_dataset = self.create_dataset(X_test, y_test)
        val_dataset = self.create_dataset(X_val, y_val) if X_val is not None else None
        
        logger.info(f"Loaded Wisconsin breast cancer dataset:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        if val_dataset:
            logger.info(f"  Val: {len(val_dataset)} samples")
        
        return train_dataset, test_dataset, val_dataset


class PhysionetLoader(TabularDataLoader):
    """Physionet 2012 Challenge dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        normalize: bool = True
    ):
        super().__init__(data_dir, cache_dir, seed, normalize)
        self.dataset_name = "physionet2012"
        self.num_classes = 2  # Mortality prediction
        self.class_names = ["survive", "die"]
        self.dataset_info = {
            "description": "ICU mortality prediction from physiological time series",
            "modality": "tabular",
            "task": "binary_classification"
        }
    
    def download_dataset(self, **kwargs) -> None:
        """Download Physionet 2012 dataset."""
        logger.warning("Physionet dataset download not implemented. Please provide data files manually.")
        logger.info("Expected files: PhysionetChallenge2012-set-a.csv")
    
    def load_and_process_physionet(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process Physionet data.
        
        Args:
            file_path: Path to Physionet CSV file
            
        Returns:
            Tuple of (features, labels)
        """
        df = pd.read_csv(file_path)
        
        # Basic preprocessing - this would need to be more sophisticated
        # for real Physionet data handling time series and missing values
        
        # Remove non-numeric columns and handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns]
        
        # Forward fill then backward fill missing values
        df_processed = df_numeric.fillna(method='ffill').fillna(method='bfill')
        
        # Separate features and labels (assuming last column is label)
        X = df_processed.iloc[:, :-1].values
        y = df_processed.iloc[:, -1].values
        
        self.feature_names = df_processed.columns[:-1].tolist()
        
        return X, y
    
    def setup_dataset(
        self,
        data_file: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup Physionet dataset."""
        if data_file is None:
            data_file = self.data_dir / "PhysionetChallenge2012-set-a.csv"
        
        if not Path(data_file).exists():
            logger.error(f"Physionet data file not found at {data_file}")
            return None, None, None
        
        # Load and process data
        X, y = self.load_and_process_physionet(data_file)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted,
                random_state=self.seed, stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        # Preprocess features
        X_train = self.preprocess_features(X_train, fit=True)
        X_test = self.preprocess_features(X_test, fit=False)
        if X_val is not None:
            X_val = self.preprocess_features(X_val, fit=False)
        
        # Preprocess labels
        y_train = self.preprocess_labels(y_train, fit=True)
        y_test = self.preprocess_labels(y_test, fit=False)
        if y_val is not None:
            y_val = self.preprocess_labels(y_val, fit=False)
        
        # Create datasets
        train_dataset = self.create_dataset(X_train, y_train)
        test_dataset = self.create_dataset(X_test, y_test)
        val_dataset = self.create_dataset(X_val, y_val) if X_val is not None else None
        
        return train_dataset, test_dataset, val_dataset


class CardiotocographyLoader(TabularDataLoader):
    """Cardiotocography (CTG) dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        normalize: bool = True
    ):
        super().__init__(data_dir, cache_dir, seed, normalize)
        self.dataset_name = "cardiotocography"
        self.num_classes = 3  # Normal, Suspect, Pathologic
        self.class_names = ["normal", "suspect", "pathologic"]
        self.dataset_info = {
            "description": "Fetal cardiotocography classification",
            "modality": "tabular",
            "task": "multiclass_classification"
        }
    
    def download_dataset(self, **kwargs) -> None:
        """Download CTG dataset from UCI."""
        import urllib.request
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls"
        save_path = self.data_dir / "CTG.xls"
        
        if not save_path.exists():
            try:
                urllib.request.urlretrieve(url, save_path)
                logger.info(f"Downloaded CTG dataset to {save_path}")
            except Exception as e:
                logger.error(f"Failed to download CTG dataset: {e}")
    
    def setup_dataset(
        self,
        data_file: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup CTG dataset."""
        if data_file is None:
            data_file = self.data_dir / "CTG.xls"
        
        if not Path(data_file).exists():
            self.download_dataset(**kwargs)
        
        if not Path(data_file).exists():
            logger.error(f"CTG data file not found at {data_file}")
            return None, None, None
        
        try:
            # Read Excel file (may require openpyxl)
            df = pd.read_excel(data_file, sheet_name="Data")
            
            # Basic preprocessing would go here
            # This is dataset-specific and would need to be implemented
            # based on the actual CTG data format
            
            logger.warning("CTG data preprocessing not fully implemented")
            return None, None, None
            
        except Exception as e:
            logger.error(f"Failed to load CTG dataset: {e}")
            return None, None, None


# Registry for tabular loaders
TABULAR_LOADERS = {
    "wisconsin_breast_cancer": WisconsinBreastCancerLoader,
    "physionet": PhysionetLoader,
    "cardiotocography": CardiotocographyLoader,
}


def get_tabular_loader(dataset_name: str, **kwargs) -> TabularDataLoader:
    """Get a tabular dataset loader by name.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Arguments passed to loader constructor
        
    Returns:
        TabularDataLoader instance
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name not in TABULAR_LOADERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(TABULAR_LOADERS.keys())}"
        )
    
    return TABULAR_LOADERS[dataset_name](**kwargs)