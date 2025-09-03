"""Base loader class for dataset loading and preprocessing."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union, List
from pathlib import Path
import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders.
    
    Provides common functionality for dataset downloading, setup, preprocessing,
    and DataLoader creation across different modalities (vision, language, tabular).
    """
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        """Initialize base loader.
        
        Args:
            data_dir: Directory to store/load datasets
            cache_dir: Directory for caching processed data
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.seed = seed
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset metadata
        self.dataset_name = None
        self.num_classes = None
        self.class_names = None
        self.dataset_info = {}
        
        # Set random seed
        torch.manual_seed(seed)
        
    @abstractmethod
    def download_dataset(self, **kwargs) -> None:
        """Download the dataset if it doesn't exist."""
        pass
    
    @abstractmethod
    def setup_dataset(
        self,
        train_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup and return train, test, validation datasets."""
        pass
    
    @abstractmethod
    def get_transforms(
        self,
        split: str = "train",
        augmentation: Optional[str] = None,
        **kwargs
    ) -> transforms.Compose:
        """Get preprocessing transforms for the given split."""
        pass
    
    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """Create a DataLoader from a dataset.
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            **kwargs: Additional DataLoader arguments
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )
    
    def balance_dataset(
        self,
        paths: List[str],
        labels: List[str],
        min_count: Optional[int] = None,
        randomize: bool = True
    ) -> Tuple[List[str], List[str]]:
        """Balance dataset by sampling equal number of samples per class.
        
        Args:
            paths: List of file paths
            labels: List of corresponding labels
            min_count: Minimum number of samples per class (None for auto)
            randomize: Whether to randomly sample or take first N samples
            
        Returns:
            Tuple of (balanced_paths, balanced_labels)
        """
        import random
        from collections import defaultdict
        
        if randomize:
            random.seed(self.seed)
        
        # Group paths by label
        label_paths = defaultdict(list)
        for path, label in zip(paths, labels):
            label_paths[label].append(path)
        
        # Determine minimum count
        if min_count is None:
            min_count = min(len(paths) for paths in label_paths.values())
        else:
            actual_min = min(len(paths) for paths in label_paths.values())
            if min_count > actual_min:
                logger.warning(
                    f"Requested min_count ({min_count}) > actual minimum ({actual_min}). "
                    f"Using actual minimum."
                )
                min_count = actual_min
        
        # Sample balanced data
        balanced_paths = []
        balanced_labels = []
        
        for label, paths in label_paths.items():
            if randomize:
                sampled_paths = random.sample(paths, min_count)
            else:
                sampled_paths = paths[:min_count]
            
            balanced_paths.extend(sampled_paths)
            balanced_labels.extend([label] * min_count)
        
        return balanced_paths, balanced_labels
    
    def balance_image_directory(
        self,
        parent_directory: Union[str, Path],
        destination_directory: Optional[Union[str, Path]] = None,
        min_count: Optional[int] = None,
        randomize: bool = True
    ) -> None:
        """Balance an image directory by class.
        
        Args:
            parent_directory: Directory containing class subdirectories
            destination_directory: If provided, copy balanced data here instead of modifying original
            min_count: Minimum samples per class
            randomize: Whether to randomly sample
        """
        parent_dir = Path(parent_directory)
        
        # Collect all image paths and labels
        paths = []
        labels = []
        print(parent_dir)
        for class_dir in parent_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for image_file in class_dir.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                        paths.append(str(image_file))
                        labels.append(class_name)
        
        # Balance the dataset
        balanced_paths, balanced_labels = self.balance_dataset(
            paths, labels, min_count, randomize
        )
        
        if destination_directory is None:
            # Remove unbalanced files from original directory
            balanced_paths_set = set(balanced_paths)
            for path in paths:
                if path not in balanced_paths_set:
                    os.remove(path)
                    logger.debug(f"Removed unbalanced file: {path}")
        else:
            # Copy balanced files to destination
            dest_dir = Path(destination_directory)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for path, label in zip(balanced_paths, balanced_labels):
                src_path = Path(path)
                dest_class_dir = dest_dir / label
                dest_class_dir.mkdir(exist_ok=True)
                
                dest_path = dest_class_dir / src_path.name
                shutil.copy2(src_path, dest_path)
                logger.debug(f"Copied balanced file: {src_path} -> {dest_path}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information including class names, counts, etc."""
        return {
            "name": self.dataset_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            **self.dataset_info
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset='{self.dataset_name}', data_dir='{self.data_dir}')"


class VisionDataLoader(BaseDataLoader):
    """Base class for vision datasets with common image preprocessing."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        image_size: int = 224
    ):
        super().__init__(data_dir, cache_dir, seed)
        self.image_size = image_size
    
    def get_default_transforms(
        self,
        split: str = "train",
        augmentation: Optional[str] = None,
        **kwargs
    ) -> List:
        """Get default image transforms for the split.
        
        Args:
            split: Dataset split ('train', 'test', 'val')
            augmentation: Augmentation type ('PatchCutout', 'Cutout', None)
            **kwargs: Additional augmentation parameters
            
        Returns:
            List of torchvision transforms
        """
        if split == "train":
            transform_list = [
                transforms.RandomRotation(7),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        
        # Add augmentation if specified
        if augmentation == "PatchCutout":
            # Import here to avoid circular imports
            try:
                from ...augmentation.patch_cutout import PatchCutout
                
                removal_fraction = kwargs.get('removal_fraction', 0.5)
                patch_size = kwargs.get('patch_size', 56)
                random_removal_fraction = kwargs.get('random_removal_fraction', False)
                random_dist = kwargs.get('random_dist', 'binomial')
                fill_val = kwargs.get('fill_val', 0)
                
                transform_list.append(
                    PatchCutout(
                        patch_height=patch_size,
                        patch_width=patch_size,
                        removal_fraction=removal_fraction,
                        random_removal_fraction=random_removal_fraction,
                        random_dist=random_dist,
                        fill_val=fill_val
                    )
                )
            except ImportError:
                logger.warning("PatchCutout not available, skipping augmentation")
                
        elif augmentation == "Cutout":
            try:
                from ...augmentation.cutout import Cutout
                n_holes = kwargs.get('n_holes', 5)
                length = kwargs.get('length', 32)
                transform_list.append(Cutout(n_holes=n_holes, length=length))
            except ImportError:
                logger.warning("Cutout not available, skipping augmentation")
        
        return transform_list
    
    def get_transforms(
        self,
        split: str = "train",
        augmentation: Optional[str] = None,
        **kwargs
    ) -> transforms.Compose:
        """Get composed transforms for the split."""
        transform_list = self.get_default_transforms(split, augmentation, **kwargs)
        return transforms.Compose(transform_list)