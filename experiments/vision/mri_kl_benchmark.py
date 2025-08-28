#!/usr/bin/env python3
"""
MRI KL Divergence Benchmark - MCal Implementation (Version 2)

Similar to experiments/get_benchmarks.py, this script provides configurable methods
for comparing KL divergence results with different calibration transforms.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from tabulate import tabulate
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import pdb

# Add MCal to path (file is now in experiments/vision/)
mcal_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcal_root))
sys.path.insert(0, str(mcal_root / "configs"))
sys.path.insert(0, str(mcal_root / "src"))

# Add XAI_Benchmark to path for data loading and augmentation
xai_root = mcal_root.parent / "XAI_Benchmark"
sys.path.insert(0, str(xai_root))
sys.path.insert(0, str(xai_root / "augmentation"))

from configs.model_dict import get_model_path
from configs.dataset_configs import get_dataset_config
import timm

# Import MCal data loaders
from src.data.loaders import mri_full_setup

# Import XAI_Benchmark augmentation utilities
from PatchCutout import PatchCutout

# Import utils directly to avoid circular imports
sys.path.insert(0, str(mcal_root / "src" / "utils"))
from optimization import get_expectation, make_one_hot, kl_divergence

# Import calibrator modules
from calibrators import MCal, MCal_CE, PlattCalibrator, TemperatureScaling

# Import transform modules for backward compatibility  
from transforms.lambda_transforms import ExpectationLambdaTransform, OptimizedLambdaTransform
from transforms.logits import LogitsSharpTransform


def load_mri_model(augmentation='vanilla', device=None):
    """Load MRI model for generating predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get model path and config
    model_path = get_model_path('mri', augmentation)
    config = get_dataset_config('mri')
    
    # Fix the class number issue - inspect actual model
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        actual_num_classes = state_dict['head.weight'].shape[0]
        print(f"Model has {actual_num_classes} classes (config says {config['num_classes']})")
    except:
        actual_num_classes = config['num_classes']
    
    # Create and load model
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=actual_num_classes)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, actual_num_classes


def load_patchcutout_predictions(dataset_type='mri', run_id=0, data_dir="../../../XAI_Benchmark/dataset_store/model_outputs"):
    """Load pre-computed PatchCutout predictions from the XAI_Benchmark format."""
    predictions_path = f"{data_dir}/{dataset_type}/PatchCutout/predictions_augmented_train_{run_id}.npy"
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"PatchCutout predictions not found at {predictions_path}. "
                              f"Please run XAI_Benchmark generation first with PatchCutout models.")
    
    print(f"Loading PatchCutout predictions from {predictions_path}")
    predictions = np.load(predictions_path)
    print(f"Loaded PatchCutout predictions with shape: {predictions.shape}")
    
    return predictions


def load_patch_drop_predictions(dataset_type='mri', run_id=0, data_dir="../../../XAI_Benchmark/dataset_store/model_outputs", fill_value=0):
    """Load pre-computed patch drop predictions from the XAI_Benchmark format."""
    predictions_path = f"{data_dir}/{dataset_type}/patch_drop_fill_{fill_value}/predictions_patch_drop_{run_id}.npy"
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Patch drop predictions not found at {predictions_path}. "
                              f"Please run XAI_Benchmark generation first with patch drop models.")
    
    print(f"Loading patch drop predictions from {predictions_path}")
    predictions = np.load(predictions_path)
    print(f"Loaded patch drop predictions with shape: {predictions.shape}")
    
    return predictions


def load_mri_dataset(batch_size=32, train_augmentation=None, **kwargs):
    """Load MRI dataset using the updated MCal data loader that mimics XAI_Benchmark exactly."""
    # Use the mri_full_setup function that now mimics XAI_Benchmark exactly
    train_dataset, test_dataset = mri_full_setup(
        train_augmentation=train_augmentation,
        **kwargs
    )
    
    # Use test dataset for consistency (it has no augmentation by default)
    dataset = test_dataset if test_dataset is not None else train_dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"Loaded MRI dataset with {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")
    
    return dataloader, dataset.classes


def generate_fractionwise_predictions_from_images(model, dataloader, n_samples=1000, n_fractions=16, 
                                                 device=None, cache_dir="./cache", use_cache=True,
                                                 use_default_data=True, return_labels=False):
    """Generate fractionwise predictions using real MRI images with PatchCutout augmentation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to use default data from XAI_Benchmark first
    if use_default_data:
        try:
            from src.data.default_data import load_default_mri_predictions
            default_predictions = load_default_mri_predictions()
            if default_predictions is not None:
                print(f"Using default MRI predictions from XAI_Benchmark with shape: {default_predictions.shape}")
                # Check if the shape matches what we expect
                if default_predictions.shape[0] == n_fractions:
                    if return_labels:
                        print("Warning: Default data doesn't include labels, falling back to generation...")
                    else:
                        return default_predictions
                else:
                    print(f"Default data has {default_predictions.shape[0]} fractions, requested {n_fractions}")
                    print("Falling back to generating new predictions...")
        except ImportError as e:
            print(f"Could not import default data loader: {e}")
        except Exception as e:
            print(f"Error loading default data: {e}")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"mri_fractionwise_predictions_{n_samples}_{n_fractions}.npy")
    
    # Check if cached predictions exist
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached predictions from {cache_file}")
        if return_labels:
            print("Warning: Cached data doesn't include labels, falling back to generation...")
        else:
            return np.load(cache_file)
    
    print(f"Generating {n_fractions} fractions with {n_samples} samples each using real MRI images...")
    
    # Get number of classes from model
    num_classes = model.head.out_features
    
    # Initialize output array: (n_fractions, n_samples, n_classes)
    all_predictions = np.zeros((n_fractions, n_samples, num_classes))
    all_labels = np.zeros(n_samples, dtype=np.int64) if return_labels else None
    
    # Define fractions (0 to 1, where 0 = no removal, 1 = full removal)
    fractions = np.linspace(0, 1, n_fractions)
    
    # Create iterator over dataloader that can be reused
    def get_batch_iterator():
        while True:
            for batch in dataloader:
                images, labels = batch
                for img, label in zip(images, labels):
                    yield img, label
    
    batch_iterator = get_batch_iterator()
    
    with torch.no_grad():
        for fraction_idx, removal_fraction in enumerate(tqdm(fractions, desc="Processing fractions")):
            # Create PatchCutout transform for this fraction
            patch_cutout = PatchCutout(
                patch_height=16,  # Standard ViT patch size
                patch_width=16,
                removal_fraction=removal_fraction,
                random_removal_fraction=False,
                fill_val=(0, 0, 0)  # Black fill for removed patches
            )
            
            fraction_predictions = []
            samples_collected = 0
            
            # Generate predictions for this fraction
            while samples_collected < n_samples:
                try:
                    # Get next image and label
                    original_img, label = next(batch_iterator)
                    
                    # Apply patch cutout augmentation
                    augmented_img = patch_cutout(original_img)
                    
                    # Add batch dimension and move to device
                    img_batch = augmented_img.unsqueeze(0).to(device)
                    
                    # Forward pass
                    logits = model(img_batch)
                    probabilities = F.softmax(logits, dim=1)
                    
                    # Store prediction
                    fraction_predictions.append(probabilities.cpu().numpy()[0])
                    
                    # Store label (only for first fraction to avoid duplication)
                    if return_labels and fraction_idx == 0:
                        all_labels[samples_collected] = label.item()
                    
                    samples_collected += 1
                    
                except StopIteration:
                    # Reset iterator if we run out of data
                    batch_iterator = get_batch_iterator()
                    continue
            
            # Convert to numpy array and store
            all_predictions[fraction_idx] = np.array(fraction_predictions)
            
            print(f"Fraction {fraction_idx+1}/{n_fractions} (removal={removal_fraction:.3f}) - Generated {samples_collected} predictions")
    
    # Save to cache
    print(f"Saving predictions to cache: {cache_file}")
    np.save(cache_file, all_predictions)
    
    if return_labels:
        return all_predictions, all_labels
    else:
        return all_predictions


def calculate_kl_metrics(outputs, device=None):
    """Calculate KL divergence metrics for outputs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_fractions, n_samples, n_outputs = outputs.shape
    
    # Results storage
    kl_values_argmax = []
    kl_values_prob = []
    
    for fraction in range(n_fractions):
        fraction_preds = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        
        # Get expectations
        one_hot_expectation, prob_expectation = get_expectation(fraction_preds, device)
        
        # Uniform distribution for comparison
        uniform_dist = torch.ones(n_outputs, device=device) / n_outputs
        
        # Calculate KL divergences
        kl_argmax = kl_divergence(one_hot_expectation, uniform_dist).item()
        kl_prob = kl_divergence(prob_expectation, uniform_dist).item()
        
        kl_values_argmax.append(kl_argmax)
        kl_values_prob.append(kl_prob)
    
    # Calculate averages
    avg_kl_argmax = np.mean(kl_values_argmax)
    avg_kl_prob = np.mean(kl_values_prob)
    
    return {
        'average_kl_argmax': avg_kl_argmax,
        'average_kl_prob': avg_kl_prob,
        'kl_values_argmax': kl_values_argmax,
        'kl_values_prob': kl_values_prob
    }


def apply_transform(outputs, method, device=None, **kwargs):
    """Apply a transformation method to the outputs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Applying {method} transform...")
    
    if method == 'baseline':
        # No transformation
        return outputs
    
    elif method == 'patchcutout':
        # PatchCutout uses pre-trained model predictions, no additional transformation needed
        return outputs
    
    elif method == 'patch_drop':
        # Patch drop uses pre-generated predictions, no additional transformation needed
        return outputs
    
    elif method == 'mcal':
        return apply_mcal_calibrator(outputs, device, **kwargs)
    
    elif method == 'mcal_ce':
        return apply_mcal_ce_calibrator(outputs, device, **kwargs)
    
    elif method == 'platt':
        return apply_platt_calibrator(outputs, device, **kwargs)
    
    elif method == 'temperature':
        return apply_temperature_calibrator(outputs, device, **kwargs)
    
    # Keep the old transform methods for backward compatibility
    elif method == 'expectation_prob':
        return apply_expectation_prob_transform(outputs, device)
    
    elif method == 'expectation_onehot':
        return apply_expectation_onehot_transform(outputs, device)
    
    elif method == 'optimized_lambda':
        return apply_optimized_lambda_transform(outputs, device, **kwargs)
    
    elif method == 'logits_sharp':
        return apply_logits_sharp_transform(outputs, device, **kwargs)
    
    else:
        raise ValueError(f"Unknown transform method: {method}")


def apply_expectation_prob_transform(outputs, device):
    """Apply probability-based expectation lambda transform using MCal module."""
    # Create temporary file for fitting
    temp_path = "/tmp/mri_temp_predictions_expectation_prob.npy"
    np.save(temp_path, outputs)
    
    # Create and fit transform
    transform = ExpectationLambdaTransform(device=device, method='prob')
    transform.fit(temp_path)
    
    # Apply transform fraction by fraction using the fitted parameters
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)
    
    for fraction in tqdm(range(n_fractions), desc="Applying expectation prob transform"):
        fraction_preds = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        
        # Get the lambdas for this fraction
        fraction_lambdas = torch.tensor(transform.lambdas[fraction], device=device)
        
        # Apply lambda adjustment manually
        adjusted_probs = fraction_preds * fraction_lambdas.unsqueeze(0)
        # Don't normalize as per original expectation lambda transform
        transformed_outputs[fraction] = adjusted_probs.cpu().numpy()
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return transformed_outputs


def apply_expectation_onehot_transform(outputs, device):
    """Apply one-hot-based expectation lambda transform using MCal module."""
    # Create temporary file for fitting
    temp_path = "/tmp/mri_temp_predictions_expectation_onehot.npy"
    np.save(temp_path, outputs)
    
    # Create and fit transform
    transform = ExpectationLambdaTransform(device=device, method='onehot')
    transform.fit(temp_path)
    
    # Apply transform fraction by fraction using the fitted parameters
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)
    
    for fraction in tqdm(range(n_fractions), desc="Applying expectation onehot transform"):
        fraction_preds = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        
        # Get the lambdas for this fraction
        fraction_lambdas = torch.tensor(transform.lambdas[fraction], device=device)
        
        # Apply lambda adjustment manually
        adjusted_probs = fraction_preds * fraction_lambdas.unsqueeze(0)
        # Don't normalize as per original expectation lambda transform
        transformed_outputs[fraction] = adjusted_probs.cpu().numpy()
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return transformed_outputs


def apply_optimized_lambda_transform(outputs, device, num_epochs=1000):
    """Apply optimized lambda transform using MCal module."""
    # Create temporary file for fitting
    temp_path = "/tmp/mri_temp_predictions_optimized.npy"
    np.save(temp_path, outputs)
    
    # Create and fit transform
    transform = OptimizedLambdaTransform(device=device)
    transform.fit(temp_path, num_epochs=num_epochs)
    
    # Apply transform fraction by fraction using the fitted parameters
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)
    
    for fraction in tqdm(range(n_fractions), desc="Applying optimized lambda transform"):
        fraction_preds = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        
        # Get the lambdas for this fraction
        fraction_lambdas = torch.tensor(transform.lambdas[fraction], device=device)
        
        # Apply lambda adjustment with normalization (as per optimized lambda transform)
        adjusted_probs = fraction_preds * fraction_lambdas.unsqueeze(0)
        adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
        transformed_outputs[fraction] = adjusted_probs.cpu().numpy()
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return transformed_outputs


def apply_logits_sharp_transform(outputs, device, num_epochs=1000, **kwargs):
    """Apply LogitsSharp transform using MCal module."""
    # Create temporary file for fitting
    temp_path = "/tmp/mri_temp_predictions_logits_sharp.npy"
    np.save(temp_path, outputs)
    
    # Create and fit transform
    transform = LogitsSharpTransform(device=device)
    transform.fit(temp_path, num_epochs=num_epochs)
    
    # Apply transform fraction by fraction using the fitted parameters  
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)
    
    for fraction in tqdm(range(n_fractions), desc="Applying logits sharp transform"):
        fraction_preds = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        
        # Apply logits sharp transformation for this fraction
        transformed_probs = transform.transform(fraction_preds, fraction_idx=fraction)
        transformed_outputs[fraction] = transformed_probs.cpu().numpy()
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return transformed_outputs


def apply_mcal_calibrator(outputs, device, kappa=4.0, max_steps=10000, **kwargs):
    """Apply MCal calibrator using uniform target distribution - single training like LogitsSharp."""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)
    
    # Create uniform target distribution
    uniform_target = torch.ones(n_classes, device=device) / n_classes
    
    # Train one MCal calibrator per fraction (like LogitsSharp trains per fraction)
    calibrators = []
    
    for fraction in tqdm(range(n_fractions), desc="Training MCal calibrators"):
        # Use current fraction as ablated probabilities
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        
        # Create and fit MCal calibrator with uniform target
        calibrator = MCal(num_classes=n_classes, target_distribution=uniform_target)
        calibrator.to(device)
        # pdb.set_trace()
        calibrator.fit(
            ablated_probs=ablated_probs,
            target_distribution=uniform_target,
            kappa=kappa,
            max_steps=max_steps,
            lr=1e-1,  # Explicitly set lr to match LogitsSharp
            verbose=False  # Disable verbose for cleaner output
        )
        calibrators.append(calibrator)
    
    # Apply calibration using trained calibrators
    for fraction in tqdm(range(n_fractions), desc="Applying MCal calibration"):
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        calibrated_probs = calibrators[fraction].forward(ablated_probs)
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()
    
    return transformed_outputs


def apply_mcal_ce_calibrator(outputs, device, max_steps=5000, head_type="linear", experiment_id="mri_experiment", **kwargs):
    """Apply MCal_CE calibrator using cross-entropy loss with target labels from 0th index (unablated predictions)."""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)
    
    # Convert outputs to tensor
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32, device=device)
    
    # Extract target labels from 0th index (unablated predictions) - same approach as MCal
    # Try to use default predictions from XAI_Benchmark first
    target_labels = None
    try:
        from src.data.default_data import load_default_mri_predictions
        default_predictions = load_default_mri_predictions()
        if default_predictions is not None:
            print(f"Using default MRI predictions from XAI_Benchmark with shape: {default_predictions.shape}")
            # Use 0th index predictions (unablated) and get argmax as target labels
            unablated_preds = torch.tensor(default_predictions[0], dtype=torch.float32, device=device)
            target_labels = unablated_preds.argmax(dim=1)
            print("Target labels extracted from default XAI_Benchmark 0th index predictions")
    except Exception as e:
        print(f"Could not load default XAI_Benchmark data: {e}")
    
    # Fallback: Use 0th index from input outputs
    if target_labels is None:
        target_labels = outputs_tensor[0].argmax(dim=1)
        print("Target labels extracted from input outputs 0th index")
    
    for fraction in tqdm(range(n_fractions), desc="Applying MCal_CE calibrator"):
        # Create and fit MCal_CE calibrator
        calibrator = MCal_CE(num_classes=n_classes, head_type=head_type)
        calibrator.to(device)
        calibrator.fit(
            ablated_probs=outputs_tensor[fraction],  # Pass current fraction (2D)
            target_labels=target_labels,  # Use target labels from 0th index
            max_steps=max_steps,
            lr=1e-3,
            verbose=True,  # Enable verbose output to match MCal behavior
            fraction=fraction,  # Pass current fraction number
            experiment_id=experiment_id  # Pass experiment identifier
        )
        
        # Apply calibration using forward method
        calibrated_probs = calibrator.forward(outputs_tensor[fraction])
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()
    
    # Combine all fraction results into a single JSON file
    print(f"\n=== Combining MCal_CE results for experiment: {experiment_id} ===")
    combined_file = MCal_CE.combine_fraction_results(experiment_id, cleanup_temp_files=True)
    if combined_file:
        print(f"All MCal_CE results combined and saved to: {combined_file}")
    else:
        print("No temporary files found to combine!")
    
    return transformed_outputs


def apply_platt_calibrator(outputs, device, max_steps=1000, **kwargs):
    """Apply Platt scaling calibrator with uniform target."""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)
    
    # Create uniform target distribution
    uniform_target = torch.ones(n_classes, device=device) / n_classes
    
    for fraction in tqdm(range(n_fractions), desc="Applying Platt calibrator"):
        # Use current fraction as ablated probabilities
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        
        # Create and fit Platt calibrator with uniform target
        calibrator = PlattCalibrator(num_classes=n_classes, target_distribution=uniform_target)
        calibrator.to(device)
        calibrator.fit(
            ablated_probs=ablated_probs,
            target_distribution=uniform_target,
            max_steps=max_steps,
            verbose=False
        )
        
        # Apply calibration using forward method
        calibrated_probs = calibrator.forward(ablated_probs)
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()
    
    return transformed_outputs


def apply_temperature_calibrator(outputs, device, max_steps=1000, **kwargs):
    """Apply temperature scaling calibrator with uniform target."""
    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)
    
    # Create uniform target distribution
    uniform_target = torch.ones(n_classes, device=device) / n_classes
    
    for fraction in tqdm(range(n_fractions), desc="Applying Temperature calibrator"):
        # Use current fraction as ablated probabilities
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        
        # Create and fit temperature scaling calibrator with uniform target
        calibrator = TemperatureScaling(num_classes=n_classes, target_distribution=uniform_target)
        calibrator.to(device)
        calibrator.fit(
            ablated_probs=ablated_probs,
            target_distribution=uniform_target,
            max_steps=max_steps,
            verbose=False
        )
        
        # Apply calibration using forward method
        calibrated_probs = calibrator.forward(ablated_probs)
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()
    
    return transformed_outputs


def aggregate_fractionwise_kl(fractionwise_results):
    """Aggregate fractionwise KL divergence results across multiple runs."""
    if not fractionwise_results or not fractionwise_results[0]:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": []}
    
    # Determine number of fractions from the first result
    first_result = fractionwise_results[0]
    if isinstance(first_result, dict) and 'kl_values_argmax' in first_result:
        num_fractions = len(first_result['kl_values_argmax'])
    else:
        return {"mean_argmax": [], "std_argmax": [], "mean_prob": [], "std_prob": []}
    
    # Initialize arrays to store values for each fraction across runs
    kl_argmax_values = [[] for _ in range(num_fractions)]
    kl_prob_values = [[] for _ in range(num_fractions)]
    
    # Collect values across all runs
    for run_results in fractionwise_results:
        kl_argmax_list = run_results['kl_values_argmax']
        kl_prob_list = run_results['kl_values_prob']
        
        for i in range(min(len(kl_argmax_list), num_fractions)):
            kl_argmax_values[i].append(kl_argmax_list[i])
            kl_prob_values[i].append(kl_prob_list[i])
    
    # Calculate mean and standard deviation for each fraction
    mean_argmax = [np.mean(values) if values else 0.0 for values in kl_argmax_values]
    std_argmax = [np.std(values) if len(values) > 1 else 0.0 for values in kl_argmax_values]
    mean_prob = [np.mean(values) if values else 0.0 for values in kl_prob_values]
    std_prob = [np.std(values) if len(values) > 1 else 0.0 for values in kl_prob_values]
    
    return {
        "mean_argmax": mean_argmax,
        "std_argmax": std_argmax,
        "mean_prob": mean_prob,
        "std_prob": std_prob
    }


def aggregate_results(all_results):
    """Aggregate results across multiple runs."""
    aggregated_results = {}
    
    for method, results in all_results.items():
        if not results:
            continue
        
        num_runs = len(results)
        
        # Extract values across runs
        kl_prob_values = [r['average_kl_prob'] for r in results]
        kl_argmax_values = [r['average_kl_argmax'] for r in results]
        
        # Aggregate fraction-wise results
        fraction_wise_results = aggregate_fractionwise_kl(results)
        
        aggregated_results[method] = {
            'kl_transformed_mean_prob': np.mean(kl_prob_values),
            'kl_transformed_std_prob': np.std(kl_prob_values),
            'kl_transformed_mean_onehot': np.mean(kl_argmax_values),
            'kl_transformed_std_onehot': np.std(kl_argmax_values),
            'fraction_wise_results_transformed': fraction_wise_results
        }
        
        # For baseline, also store as baseline results
        if method == 'baseline':
            aggregated_results['baseline'] = {
                'kl_baseline_mean_prob': np.mean(kl_prob_values),
                'kl_baseline_std_prob': np.std(kl_prob_values),
                'kl_baseline_mean_onehot': np.mean(kl_argmax_values),
                'kl_baseline_std_onehot': np.std(kl_argmax_values),
                'fraction_wise_results': fraction_wise_results
            }
    
    return aggregated_results


def build_kl_comparison_table(aggregated_results, include_methods=None):
    """Build comparison table for KL divergence results."""
    # Initialize table data
    table_data = [["Method", "Average KL (Prob)", "Average KL (Argmax)"]]
    
    # Define method display names
    method_names = {
        'baseline': "Original",
        'patchcutout': "PatchCutout-trained Model",
        'patch_drop': "Patch Dropping",
        'mcal': "MCal (Vector Scaling)",
        'mcal_ce': "MCal_CE (Cross-Entropy)",
        'platt': "Platt Scaling",
        'temperature': "Temperature Scaling",
        'logits_sharp': "Logits Sharp Transform",
        # Keep old transform names for backward compatibility
        'expectation_prob': "Expectation Probability Transform",
        'expectation_onehot': "Expectation One-hot Transform", 
        'optimized_lambda': "Optimized Lambda Transform"
    }
    
    # Add baseline if available
    if 'baseline' in aggregated_results:
        baseline = aggregated_results['baseline']
        table_data.append([
            method_names['baseline'], 
            f"{baseline['kl_baseline_mean_prob']:.2e} ± {baseline['kl_baseline_std_prob']:.2e}", 
            f"{baseline['kl_baseline_mean_onehot']:.2e} ± {baseline['kl_baseline_std_onehot']:.2e}"
        ])
    
    # Methods to include in the table
    methods_to_include = include_methods or [m for m in aggregated_results.keys() if m != 'baseline']
    
    # Add results for each method
    for method in methods_to_include:
        if method not in aggregated_results or method == 'baseline':
            continue
            
        result = aggregated_results[method]
        if 'kl_transformed_mean_prob' in result:
            table_data.append([
                method_names.get(method, method.replace('_', ' ').title()), 
                f"{result['kl_transformed_mean_prob']:.2e} ± {result['kl_transformed_std_prob']:.2e}", 
                f"{result['kl_transformed_mean_onehot']:.2e} ± {result['kl_transformed_std_onehot']:.2e}"
            ])
    
    # Generate table
    table = tabulate(table_data, headers="firstrow", tablefmt="grid")
    return table


def process_mri_dataset(methods=None, device="cuda", save_dir="./results", n_runs=3, 
                       n_samples=1000, n_fractions=16, overwrite=False, use_cache=True,
                       patchcutout_data_dir="./dataset_store/model_outputs", use_default_data=True):
    """
    Process MRI dataset and generate benchmarks with multiple runs.
    
    Args:
        methods (list): List of methods to use
        device (str): Device to use for computation
        save_dir (str): Directory to save results
        n_runs (int): Number of runs for each method
        n_samples (int): Number of samples per fraction
        n_fractions (int): Number of fractions to generate
        overwrite (bool): Whether to overwrite existing results
        use_cache (bool): Whether to use cached predictions for vanilla model
        patchcutout_data_dir (str): Directory containing PatchCutout predictions
        use_default_data (bool): Whether to use default MRI data from XAI_Benchmark
    """
    # Default methods - include all calibrators and pre-computed methods
    if methods is None:
        methods = ['baseline', 'patchcutout', 'patch_drop', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp']
    
    device = torch.device(device)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)
    
    print("="*60)
    print("MRI KL Divergence Benchmark")
    print("="*60)
    print(f"Methods: {methods}")
    print(f"Runs: {n_runs}")
    print(f"Samples per fraction: {n_samples}")
    print(f"Fractions: {n_fractions}")
    print(f"Device: {device}")
    
    # Load MRI model (only needed for methods that aren't pre-computed)
    model, num_classes, dataloader, class_names = None, None, None, None
    precomputed_methods = {'patchcutout', 'patch_drop'}
    
    if any(method not in precomputed_methods for method in methods):
        print("\nLoading MRI model...")
        model, num_classes = load_mri_model('vanilla', device)
        print(f"Loaded model with {num_classes} classes")
        
        # Load MRI dataset using our updated loader
        print("\nLoading MRI dataset using MCal loader (XAI_Benchmark compatible)...")
        dataloader, class_names = load_mri_dataset()
        print(f"Dataset classes: {class_names}")
    
    # Initialize results storage
    all_results = {method: [] for method in methods}
    
    # Run multiple experiments
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        # Process each method
        for method in methods:
            print(f"\nProcessing method: {method}")
            
            if method == 'patchcutout':
                # Load PatchCutout predictions
                try:
                    predictions = load_patchcutout_predictions(
                        dataset_type='mri', 
                        run_id=run, 
                        data_dir=patchcutout_data_dir
                    )
                    print(f"Using PatchCutout predictions with shape: {predictions.shape}")
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    print("Skipping PatchCutout method for this run.")
                    continue
            elif method == 'patch_drop':
                # Load patch drop predictions
                try:
                    predictions = load_patch_drop_predictions(
                        dataset_type='mri', 
                        run_id=run, 
                        data_dir=patchcutout_data_dir,
                        fill_value=0  # Use 0 as default fill value
                    )
                    print(f"Using patch drop predictions with shape: {predictions.shape}")
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    print("Skipping patch drop method for this run.")
                    continue
            else:
                # Generate baseline predictions from real images for other methods
                if model is None:
                    print("Error: Model not loaded for non-PatchCutout methods")
                    continue
                
                # MCal_CE now handles labels internally, no need to request them
                need_labels = False
                    
                result = generate_fractionwise_predictions_from_images(
                    model, dataloader, n_samples, n_fractions, device, 
                    cache_dir=os.path.join(save_dir, "cache"), use_cache=use_cache,
                    use_default_data=use_default_data, return_labels=need_labels
                )
                
                if need_labels:
                    predictions, target_labels = result
                else:
                    predictions = result
                    target_labels = None
            
            # Apply transformation
            if method in ['baseline', 'patchcutout', 'patch_drop']:
                transformed_predictions = predictions
            else:
                # Configure method-specific parameters
                method_kwargs = {}
                if method in ['mcal', 'mcal_ce', 'platt', 'temperature']:
                    method_kwargs['max_steps'] = 1000  # Calibrator optimization steps (match MCal default for full convergence)
                elif method == 'mcal':
                    method_kwargs['kappa'] = 10.0  # Sharpening parameter for MCal
                elif method == 'mcal_ce':
                    method_kwargs['max_steps'] = 5000  # More steps for cross-entropy training
                    method_kwargs['head_type'] = 'linear'  # Use linear head by default
                    # MCal_CE now handles target labels internally from 0th index predictions
                elif method == 'optimized_lambda':
                    method_kwargs['num_epochs'] = 500  # Reduced for demo
                
                transformed_predictions = apply_transform(
                    predictions, method, device, **method_kwargs
                )
            
            # Calculate KL metrics
            kl_results = calculate_kl_metrics(transformed_predictions, device)
            all_results[method].append(kl_results)
            
            print(f"  KL (prob): {kl_results['average_kl_prob']:.6f}")
            print(f"  KL (argmax): {kl_results['average_kl_argmax']:.6f}")
    
    # Aggregate results
    print("\nAggregating results across all runs...")
    aggregated_results = aggregate_results(all_results)
    
    # Save results as JSON
    json_path = os.path.join(save_dir, "json", "aggregated_results_mri.json")
    
    # Convert to JSON serializable format
    json_serializable_results = convert_to_json_serializable(aggregated_results)
    
    with open(json_path, 'w') as f:
        json.dump(json_serializable_results, f, indent=4)
    print(f"Aggregated results saved to {json_path}")
    
    # Build and display comparison table
    table = build_kl_comparison_table(aggregated_results, include_methods=methods)
    
    print(f"\nKL Divergence Comparison for MRI (averaged over {n_runs} runs):")
    print(table)
    
    # Save table
    table_path = os.path.join(save_dir, "kl_comparison_table_mri.txt")
    with open(table_path, 'w') as f:
        f.write(f"KL Divergence Comparison for MRI (averaged over {n_runs} runs):\n")
        f.write(table)
    print(f"Comparison table saved to {table_path}")
    
    return aggregated_results


def convert_to_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON serializable types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def test_mri_data_generation_consistency():
    """Test that MCal MRI data generation matches XAI_Benchmark exactly."""
    print("Testing MRI data generation consistency...")
    
    try:
        # Test basic dataset loading
        print("1. Testing basic dataset loading...")
        train_dataset, test_dataset = mri_full_setup()
        print(f"   Train dataset: {len(train_dataset) if train_dataset else 'None'} samples")
        print(f"   Test dataset: {len(test_dataset) if test_dataset else 'None'} samples")
        
        # Test with PatchCutout augmentation
        print("2. Testing PatchCutout augmentation...")
        train_aug_dataset, test_aug_dataset = mri_full_setup(
            train_augmentation="PatchCutout",
            removal_fraction=0.5,
            patch_size=56,
            fill_val=(0, 0, 0)
        )
        print(f"   Train dataset (with PatchCutout): {len(train_aug_dataset) if train_aug_dataset else 'None'} samples")
        print(f"   Test dataset (with PatchCutout): {len(test_aug_dataset) if test_aug_dataset else 'None'} samples")
        
        # Test with Cutout augmentation  
        print("3. Testing Cutout augmentation...")
        train_cutout_dataset, test_cutout_dataset = mri_full_setup(
            train_augmentation="Cutout"
        )
        print(f"   Train dataset (with Cutout): {len(train_cutout_dataset) if train_cutout_dataset else 'None'} samples")
        print(f"   Test dataset (with Cutout): {len(test_cutout_dataset) if test_cutout_dataset else 'None'} samples")
        
        # Check class consistency
        if train_dataset and test_dataset:
            print("4. Checking class consistency...")
            print(f"   Train classes: {train_dataset.classes}")
            print(f"   Test classes: {test_dataset.classes}")
            print(f"   Classes match: {train_dataset.classes == test_dataset.classes}")
        
        print("\n✓ MRI data generation consistency test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ MRI data generation consistency test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MRI KL Divergence Benchmark")
    parser.add_argument("--methods", nargs='+', 
                       default=['baseline', 'patchcutout', 'patch_drop', 'mcal', 'mcal_ce', 'platt', 'temperature', 'logits_sharp'],
                       help="Methods to include in benchmark. Available: baseline, patchcutout, patch_drop, mcal, mcal_ce, platt, temperature, logits_sharp, expectation_prob, expectation_onehot, optimized_lambda")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per fraction")
    parser.add_argument("--fractions", type=int, default=16, help="Number of fractions")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="./results", help="Save directory")
    parser.add_argument("--patchcutout_data_dir", type=str, default="../../../XAI_Benchmark/dataset_store/model_outputs", 
                       help="Directory containing PatchCutout predictions from XAI_Benchmark")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of generated predictions")
    parser.add_argument("--no-default-data", action="store_true", help="Disable use of default MRI data from XAI_Benchmark")
    parser.add_argument("--test-data-consistency", action="store_true", 
                       help="Test that MRI data generation matches XAI_Benchmark exactly")
    
    args = parser.parse_args()
    
    # Run data consistency test if requested
    if args.test_data_consistency:
        print("Running MRI data generation consistency test...")
        success = test_mri_data_generation_consistency()
        if success:
            print("✓ Data generation consistency test passed!")
        else:
            print("✗ Data generation consistency test failed!")
            sys.exit(1)
        return
    
    # Set device
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {device}")
    
    # pdb.set_trace()
    # Run benchmark
    aggregated_results = process_mri_dataset(
        methods=args.methods,
        device=device,
        save_dir=args.save_dir,
        n_runs=args.runs,
        n_samples=args.samples, 
        n_fractions=args.fractions,
        overwrite=args.overwrite,
        use_cache=not args.no_cache,
        patchcutout_data_dir=args.patchcutout_data_dir,
        use_default_data=not args.no_default_data
    )
    
    print("\nBenchmark completed! YAY!")


if __name__ == "__main__":
    # pdb.set_trace()
    main()