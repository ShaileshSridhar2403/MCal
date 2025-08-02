"""Language dataset loaders for medical Q&A and other text datasets."""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List
import logging

import torch
from torch.utils.data import Dataset

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class LanguageDataset(Dataset):
    """Generic language dataset class."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 512
    ):
        """Initialize language dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels (optional for inference)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        
        # Tokenize if tokenizer is provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'text': text
            }
        else:
            item = {'text': text}
        
        # Add label if available
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class MedicalQALoader(BaseDataLoader):
    """Base class for medical Q&A dataset loaders."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        super().__init__(data_dir, cache_dir, seed)
        self.tokenizer = None
        self.max_length = 512
    
    def set_tokenizer(self, tokenizer, max_length: int = 512) -> None:
        """Set tokenizer for text processing.
        
        Args:
            tokenizer: HuggingFace tokenizer or similar
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def create_dataset(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None
    ) -> LanguageDataset:
        """Create a language dataset from texts and labels."""
        return LanguageDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )


class MedMCQALoader(MedicalQALoader):
    """MedMCQA medical multiple choice Q&A dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        super().__init__(data_dir, cache_dir, seed)
        self.dataset_name = "medmcqa"
        self.num_classes = 4  # Multiple choice A, B, C, D
        self.class_names = ["A", "B", "C", "D"]
        self.dataset_info = {
            "description": "Medical multiple choice question answering",
            "modality": "language",
            "task": "multiple_choice_qa"
        }
    
    def download_dataset(self, **kwargs) -> None:
        """Download MedMCQA dataset."""
        try:
            from datasets import load_dataset
            
            # Download from HuggingFace
            dataset = load_dataset("medmcqa")
            
            # Save to local files
            splits = ["train", "validation", "test"]
            for split in splits:
                if split in dataset:
                    save_path = self.data_dir / f"medmcqa_{split}.json"
                    dataset[split].to_json(save_path)
                    logger.info(f"Saved MedMCQA {split} split to {save_path}")
                    
        except ImportError:
            logger.error("datasets library not found. Install with: pip install datasets")
        except Exception as e:
            logger.error(f"Failed to download MedMCQA: {e}")
    
    def load_data_from_file(self, file_path: Union[str, Path]) -> Tuple[List[str], List[int]]:
        """Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Tuple of (questions, answers)
        """
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        questions = []
        answers = []
        
        for item in data:
            # Format question with choices
            question = item['question']
            choices = [
                f"A) {item['opa']}",
                f"B) {item['opb']}",
                f"C) {item['opc']}",
                f"D) {item['opd']}"
            ]
            
            formatted_question = f"{question}\n\n" + "\n".join(choices)
            questions.append(formatted_question)
            
            # Answer is typically 1-4, convert to 0-3
            answer = item['cop'] - 1 if 'cop' in item else 0
            answers.append(answer)
        
        return questions, answers
    
    def setup_dataset(
        self,
        train_file: Optional[str] = None,
        test_file: Optional[str] = None,
        val_file: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup MedMCQA dataset.
        
        Args:
            train_file: Path to training JSON file
            test_file: Path to test JSON file
            val_file: Path to validation JSON file
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (train_dataset, test_dataset, val_dataset)
        """
        # Set default file paths
        if train_file is None:
            train_file = self.data_dir / "medmcqa_train.json"
        if test_file is None:
            test_file = self.data_dir / "medmcqa_test.json"
        if val_file is None:
            val_file = self.data_dir / "medmcqa_validation.json"
        
        # Download if files don't exist
        if not any(Path(f).exists() for f in [train_file, test_file, val_file]):
            self.download_dataset(**kwargs)
        
        train_dataset = None
        test_dataset = None
        val_dataset = None
        
        # Load datasets
        if Path(train_file).exists():
            questions, answers = self.load_data_from_file(train_file)
            train_dataset = self.create_dataset(questions, answers)
            logger.info(f"Loaded {len(questions)} training samples")
        
        if Path(test_file).exists():
            questions, answers = self.load_data_from_file(test_file)
            test_dataset = self.create_dataset(questions, answers)
            logger.info(f"Loaded {len(questions)} test samples")
        
        if Path(val_file).exists():
            questions, answers = self.load_data_from_file(val_file)
            val_dataset = self.create_dataset(questions, answers)
            logger.info(f"Loaded {len(questions)} validation samples")
        
        return train_dataset, test_dataset, val_dataset


class MedQALoader(MedicalQALoader):
    """MedQA medical question answering dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        super().__init__(data_dir, cache_dir, seed)
        self.dataset_name = "medqa"
        self.num_classes = 4  # Multiple choice A, B, C, D
        self.class_names = ["A", "B", "C", "D"]
        self.dataset_info = {
            "description": "Medical question answering dataset",
            "modality": "language",
            "task": "multiple_choice_qa"
        }
    
    def download_dataset(self, **kwargs) -> None:
        """Download MedQA dataset."""
        logger.warning("MedQA dataset download not implemented. Please provide data files manually.")
    
    def setup_dataset(
        self,
        train_file: Optional[str] = None,
        test_file: Optional[str] = None,
        val_file: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup MedQA dataset."""
        # Similar implementation to MedMCQA but with different data format
        logger.warning("MedQA loader not fully implemented")
        return None, None, None


class AI2ARCLoader(MedicalQALoader):
    """AI2 ARC (Allen Institute Reasoning Challenge) dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        super().__init__(data_dir, cache_dir, seed)
        self.dataset_name = "ai2_arc"
        self.num_classes = 4  # Typically 4 choices
        self.dataset_info = {
            "description": "Science question answering and reasoning",
            "modality": "language",
            "task": "multiple_choice_qa"
        }
    
    def download_dataset(self, challenge_set: str = "ARC-Challenge", **kwargs) -> None:
        """Download AI2 ARC dataset.
        
        Args:
            challenge_set: Either 'ARC-Challenge' or 'ARC-Easy'
            **kwargs: Additional arguments
        """
        try:
            from datasets import load_dataset
            
            # Download from HuggingFace
            dataset = load_dataset("ai2_arc", challenge_set)
            
            # Save to local files
            for split in dataset.keys():
                save_path = self.data_dir / f"ai2_arc_{challenge_set.lower()}_{split}.json"
                dataset[split].to_json(save_path)
                logger.info(f"Saved AI2 ARC {split} split to {save_path}")
                
        except ImportError:
            logger.error("datasets library not found. Install with: pip install datasets")
        except Exception as e:
            logger.error(f"Failed to download AI2 ARC: {e}")
    
    def setup_dataset(
        self,
        train_file: Optional[str] = None,
        test_file: Optional[str] = None,
        val_file: Optional[str] = None,
        challenge_set: str = "ARC-Challenge",
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup AI2 ARC dataset."""
        # Download if needed
        if not any(Path(self.data_dir).glob("ai2_arc_*.json")):
            self.download_dataset(challenge_set=challenge_set, **kwargs)
        
        # Implementation would be similar to MedMCQA
        logger.warning("AI2 ARC loader not fully implemented")
        return None, None, None


class PhysionetLoader(BaseDataLoader):
    """Physionet 2012 mortality prediction dataset loader."""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        super().__init__(data_dir, cache_dir, seed)
        self.dataset_name = "physionet"
        self.num_classes = 2  # Binary mortality prediction
        self.class_names = ["survive", "die"]
        self.dataset_info = {
            "description": "ICU mortality prediction from physiological data",
            "modality": "tabular",
            "task": "binary_classification"
        }
    
    def download_dataset(self, **kwargs) -> None:
        """Download Physionet 2012 dataset."""
        logger.warning("Physionet dataset download not implemented. Please provide CSV files manually.")
    
    def load_csv_data(self, file_path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (features, labels)
        """
        df = pd.read_csv(file_path)
        
        # Separate features and labels
        # Assuming last column is label
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def setup_dataset(
        self,
        train_file: Optional[str] = None,
        test_file: Optional[str] = None,
        val_file: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Setup Physionet dataset."""
        # Would need custom tabular dataset class
        logger.warning("Physionet loader not fully implemented - requires tabular dataset class")
        return None, None, None


# Registry for language loaders
LANGUAGE_LOADERS = {
    "medmcqa": MedMCQALoader,
    "medqa": MedQALoader,
    "ai2_arc": AI2ARCLoader,
    "physionet": PhysionetLoader,
}


def get_language_loader(dataset_name: str, **kwargs) -> BaseDataLoader:
    """Get a language dataset loader by name.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Arguments passed to loader constructor
        
    Returns:
        BaseDataLoader instance
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name not in LANGUAGE_LOADERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(LANGUAGE_LOADERS.keys())}"
        )
    
    return LANGUAGE_LOADERS[dataset_name](**kwargs)