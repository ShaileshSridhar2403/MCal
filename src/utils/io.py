"""Input/Output utilities for MCal."""

import json
import pickle
import numpy as np
import torch
from typing import Any, Dict, Union
from pathlib import Path


def save_results(results: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save results dictionary to file.
    
    Args:
        results: Dictionary containing results to save
        path: Path to save results to
    """
    path = Path(path)
    
    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.json':
        # Convert numpy arrays and tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable_results[key] = value.cpu().numpy().tolist()
            else:
                serializable_results[key] = value
                
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    else:
        # Use pickle for other formats
        with open(path, 'wb') as f:
            pickle.dump(results, f)


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load results dictionary from file.
    
    Args:
        path: Path to load results from
        
    Returns:
        Dictionary containing loaded results
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            results = json.load(f)
    else:
        with open(path, 'rb') as f:
            results = pickle.load(f)
    
    return results


def save_tensor(tensor: Union[np.ndarray, torch.Tensor], path: Union[str, Path]) -> None:
    """Save tensor to file.
    
    Args:
        tensor: Tensor to save
        path: Path to save tensor to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    np.save(path, tensor)


def load_tensor(path: Union[str, Path], device: torch.device = None) -> torch.Tensor:
    """Load tensor from file.
    
    Args:
        path: Path to load tensor from
        device: Device to load tensor on
        
    Returns:
        Loaded tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    
    array = np.load(path)
    return torch.tensor(array, dtype=torch.float32, device=device)