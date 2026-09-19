"""Vision dataset loaders for BreakHis, ChexPert, MRI, and ImageNet/ImageNette."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

from torch.utils.data import Dataset
from torchvision import datasets

from .base_loader import VisionDataLoader



logger = logging.getLogger(__name__)


class BreakHisLoader(VisionDataLoader):
    """BreakHis breast cancer histology dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        image_size: int = 224
    ):
        super().__init__(data_dir, cache_dir, seed, image_size)
        self.dataset_name = "breakhis"
        self.num_classes = 8  # 4 tumor types × 2 magnifications typically used
        self.dataset_info = {
            "description": "Breast cancer histopathological image classification",
            "modality": "vision",
            "task": "classification"
        }
    
    def download_dataset(
        self,
        source_path: Optional[str] = None,
        kaggle_credentials: Optional[str] = None,
        **kwargs
    ) -> None:
        """Download BreakHis dataset.
        
        Args:
            source_path: Path to local dataset zip file
            kaggle_credentials: Path to kaggle.json for Kaggle download
            **kwargs: Additional arguments
        """
        if source_path and Path(source_path).exists():
            # Copy from local source
            dataset_zip = self.data_dir / "dataset_v2.zip"
            if not dataset_zip.exists():
                shutil.copy(source_path, dataset_zip)
                logger.info(f"Copied dataset from {source_path}")
        else:
            # TODO: Add Kaggle download support
            logger.warning("No source path provided. Please provide dataset_v2.zip manually.")
    
    def setup_dataset(
        self,
        train_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        n_examples: Optional[int] = None,
        train_augmentation: Optional[str] = None,
        test_augmentation: Optional[str] = None,
        overwrite: bool = False,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup BreakHis dataset.
        
        Args:
            train_dir: Training data directory
            test_dir: Test data directory  
            val_dir: Validation data directory (optional)
            n_examples: Number of examples per class (None for all)
            train_augmentation: Training augmentation type
            test_augmentation: Test augmentation type
            overwrite: Whether to overwrite existing directories
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (train_dataset, test_dataset, val_dataset)
        """
        if train_dir is None:
            train_dir = str(self.data_dir / "BreakHisTraining")
        if test_dir is None:
            test_dir = str(self.data_dir / "BreakHisTesting")
            
        train_dataset = None
        test_dataset = None
        val_dataset = None
        
        # Handle overwrite
        if overwrite:
            for dir_path in [train_dir, test_dir]:
                if dir_path and os.path.exists(dir_path):
                    logger.info(f"Overwrite flag set. Removing existing directory: {dir_path}")
                    shutil.rmtree(dir_path)
        
        # Setup training data
        if train_dir is not None:
            if not os.path.exists(train_dir):
                logger.info(f"Dataset not found at {train_dir}, setting it up...")
                self._extract_dataset(train_dir, test_dir)
                self.balance_image_directory(train_dir)
            else:
                logger.info(f"Dataset already present at {train_dir}")
            
            train_transform = self.get_transforms("train", train_augmentation, **kwargs)
            train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        
        # Setup test data
        if test_dir is not None:
            if not os.path.exists(test_dir):
                logger.info(f"Test dataset not found at {test_dir}, setting it up...")
                self._extract_dataset(train_dir, test_dir)
                self.balance_image_directory(test_dir, randomize=False)
            else:
                logger.info(f"Test dataset already present at {test_dir}")
            
            test_transform = self.get_transforms("test", test_augmentation, **kwargs)
            test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        
        # Update class information
        if train_dataset is not None:
            self.num_classes = len(train_dataset.classes)
            self.class_names = train_dataset.classes
        elif test_dataset is not None:
            self.num_classes = len(test_dataset.classes)
            self.class_names = test_dataset.classes
        
        return train_dataset, test_dataset, val_dataset
    
    def _extract_dataset(self, train_dir: str, test_dir: Optional[str] = None) -> None:
        """Extract dataset from zip file."""
        dataset_zip = self.data_dir / "dataset_v2.zip"
        
        if not dataset_zip.exists():
            # Try to find in different locations
            possible_paths = [
                "./datasets/dataset_v2.zip",
                "../datasets/dataset_v2.zip",
                "../../datasets/dataset_v2.zip",
                "/home/antonxue/shailesh/MCal/data/BreakHis/dataset_v2.zip",
                "./data/BreakHis/dataset_v2.zip"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    shutil.copy(path, dataset_zip)
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find dataset_v2.zip. Please place it at {dataset_zip} "
                    "or provide source_path in download_dataset()"
                )
        
        # Extract archive
        shutil.unpack_archive(str(dataset_zip), str(self.data_dir))
        
        # Move directories to expected locations
        extracted_dir = self.data_dir / "dataset_v2"
        if (extracted_dir / "train").exists():
            shutil.move(str(extracted_dir / "train"), train_dir)
        if test_dir and (extracted_dir / "test").exists():
            shutil.move(str(extracted_dir / "test"), test_dir)
        
        # Cleanup
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)


class MRILoader(VisionDataLoader):
    """Brain tumor MRI dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        image_size: int = 224
    ):
        super().__init__(data_dir, cache_dir, seed, image_size)
        self.dataset_name = "mri"
        self.num_classes = 4  # glioma, meningioma, no_tumor, pituitary
        self.class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        self.dataset_info = {
            "description": "Brain tumor MRI classification",
            "modality": "vision",
            "task": "classification"
        }
    
    def download_dataset(
        self,
        kaggle_credentials: Optional[str] = None,
        **kwargs
    ) -> None:
        """Download MRI dataset from Kaggle.
        
        Args:
            kaggle_credentials: Path to kaggle.json file
            **kwargs: Additional arguments
        """
        logger.info("Downloading MRI Dataset....")
        if kaggle_credentials:
            self._setup_kaggle_credentials(kaggle_credentials)
        
        try:
            # Download using Kaggle API
            cmd = "kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset"
            result = subprocess.run(cmd.split(), cwd=self.data_dir, capture_output=True, text=True)
            logger.info(result)
            if result.returncode == 0:
                # Extract the downloaded zip
                zip_path = self.data_dir / "brain-tumor-mri-dataset.zip"
                if zip_path.exists():
                    shutil.unpack_archive(str(zip_path), str(self.data_dir))
                    logger.info("Downloaded and unpacked MRI dataset")
                else:
                    logger.error("Download succeeded but zip file not found")
            else:
                logger.error(f"Kaggle download failed: {result.stderr}")
                
        except FileNotFoundError:
            logger.error("Kaggle CLI not found. Please install with: pip install kaggle")
    
    def _setup_kaggle_credentials(self, credentials_path: str) -> None:
        """Setup Kaggle credentials."""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        dest_path = kaggle_dir / "kaggle.json"
        shutil.copy(credentials_path, dest_path)
        dest_path.chmod(0o600)
        logger.info("Kaggle credentials setup complete")
    
    def setup_dataset(
        self,
        train_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        n_examples: Optional[int] = None,
        train_augmentation: Optional[str] = None,
        test_augmentation: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup MRI dataset."""
        if train_dir is None:
            train_dir = str(self.data_dir / "Training")
        if test_dir is None:
            test_dir = str(self.data_dir / "Testing")
        
        train_dataset = None
        test_dataset = None
        val_dataset = None
        
        # Set default fill_val for RGB images
        kwargs.setdefault('fill_val', (0, 0, 0))
        
        # Setup training data
        if train_dir is not None:
            if not os.path.isdir(train_dir):
                logger.info(f"No dataset found at {train_dir}, proceeding to download")
                self.download_dataset(**kwargs)
                self.balance_image_directory(train_dir)
            else:
                logger.info("Existing downloaded MRI dataset found, proceeding with data processing")
            
            train_transform = self.get_transforms("train", train_augmentation, **kwargs)
            train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    

        if test_dir is not None:
            if not os.path.isdir(test_dir):
                logger.info(f"No dataset found at {test_dir}, proceeding to download")
                self.download_dataset(**kwargs)
                self.balance_image_directory(test_dir, randomize=False)
            else:
                logger.info("Existing downloaded MRI test dataset found, proceeding with data processing")
            
            test_transform = self.get_transforms("test", test_augmentation, **kwargs)
            test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        
        return train_dataset, test_dataset, val_dataset





class ChexPertLoader(VisionDataLoader):
    """CheXpert chest X-ray dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        image_size: int = 224
    ):
        super().__init__(data_dir, cache_dir, seed, image_size)
        self.dataset_name = "chexpert"
        self.num_classes = 14  # 14 pathology labels
        self.class_names = ["No Cardiomegaly","Cardiomegaly"]
        self.dataset_info = {
            "description": "Chest X-ray pathology classification",
            "modality": "vision",
            "task": "multi_label_classification"
        }
    
    def download_dataset(self, **kwargs) -> None:
        """Download CheXpert dataset."""
        logger.warning(
            "CheXpert dataset requires registration at "
            "https://stanfordmlgroup.github.io/competitions/chexpert/"
        )
        # Dataset download would require manual registration
    
    def setup_dataset(
        self,
        train_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup CheXpert dataset."""
        # This would need custom dataset class for multi-label classification
        # and CSV parsing logic
        

        logger.warning("CheXpert loader not fully implemented - requires custom dataset class")

        return None, None, None



# Registry for easy access to loaders
VISION_LOADERS = {
    "breakhis": BreakHisLoader,
    "mri": MRILoader,
    "chexpert": ChexPertLoader,
}


def get_vision_loader(dataset_name: str, **kwargs) -> VisionDataLoader:
    """Get a vision dataset loader by name.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Arguments passed to loader constructor
        
    Returns:
        VisionDataLoader instance
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name not in VISION_LOADERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(VISION_LOADERS.keys())}"
        )
    
    return VISION_LOADERS[dataset_name](**kwargs)


def mri_full_setup(
    data_dir: Optional[Union[str, Path]] = None,
    train_augmentation: Optional[str] = None,
    test_augmentation: Optional[str] = None,
    image_size: int = 224,
    seed: int = 42,
    **kwargs
) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """
    Full setup function for MRI dataset that mimics XAI_Benchmark interface.
    
    This function provides a simplified interface for MRI dataset loading
    that's compatible with existing benchmark scripts.
    
    Args:
        data_dir: Directory to store/load datasets
        train_augmentation: Training augmentation type (e.g., "PatchCutout", "Cutout")
        test_augmentation: Test augmentation type
        image_size: Image size for transforms
        seed: Random seed
        **kwargs: Additional arguments passed to transforms
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Initialize MRI loader
    loader = MRILoader(
        data_dir=data_dir,
        seed=seed,
        image_size=image_size
    )
    
    # Setup datasets with augmentation
    train_dataset, test_dataset, _ = loader.setup_dataset(
        train_augmentation=train_augmentation,
        test_augmentation=test_augmentation,
        **kwargs
    )
    
    return train_dataset, test_dataset





