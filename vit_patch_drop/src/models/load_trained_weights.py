import torch
import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from .vision_transformer import vit_base_patch16_224, _conv_filter

def load_vit_model(weights_path, num_classes=4, patch_size=16, device=None):
    """
    Load a Vision Transformer model with custom weights.
    
    Args:
        weights_path (str): Path to the model weights file
        num_classes (int): Number of output classes
        patch_size (int): Patch size for the ViT model
        device (torch.device): Device to load the model on (None for auto-detection)
        
    Returns:
        model: The loaded ViT model
        device: The device the model is loaded on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = vit_base_patch16_224(pretrained=False, num_classes=num_classes)
    
    print(f"Loading weights from: {weights_path}")
    
    # Load weights
    try:
        timm_state_dict = torch.load(weights_path, map_location=device)
        print("Successfully loaded checkpoint file")
    except FileNotFoundError:
        print(f"Error: Weight file not found at {weights_path}")
        return None, device
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None, device
    
    # Extract model weights if needed
    if 'model' in timm_state_dict:
        timm_state_dict = timm_state_dict['model']
        print("Using 'model' key from checkpoint")
    elif 'state_dict' in timm_state_dict:
        timm_state_dict = timm_state_dict['state_dict']
        print("Using 'state_dict' key from checkpoint")
    else:
        print("Using entire checkpoint as state_dict")
    
    # Handle prefix differences
    if all(k.startswith('model.') for k in timm_state_dict.keys()):
        print("Removing 'model.' prefix from keys")
        timm_state_dict = {k[6:]: v for k, v in timm_state_dict.items()}
    
    # Apply conv filter for patch embedding
    timm_state_dict = _conv_filter(timm_state_dict, patch_size=patch_size)
    
    # Load weights into model
    result = model.load_state_dict(timm_state_dict, strict=True)
    print(f"Successfully loaded weights!")
    print(f"Missing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model, device

def create_patch_mask(mask_type='all', specific_patches=None, n_patches=None, total_patches=197, batch_size=1, device=None):
    """
    Create a patch mask for ViT inference.
    
    Args:
        mask_type (str): Type of mask to create ('all', 'specific', 'first_n', or 'indices')
        specific_patches (list): List of patch indices to drop (for 'specific' type) or keep (for 'indices' type)
        n_patches (int): Number of patches to keep (for 'first_n' type)
        total_patches (int): Total number of patches including class token
        batch_size (int): Batch size for creating batch-specific masks
        device (torch.device): Device to create the mask on
        
    Returns:
        torch.Tensor: The created patch mask in one of two formats:
            - 1D Tensor: Indices of patches to keep, including class token (index 0)
            - 2D Tensor: [batch_size, num_patches_to_keep] for batch-specific masks
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if mask_type == 'all':
        # Keep all patches - return indices of all patches (0 to total_patches-1)
        return torch.arange(total_patches).long().to(device)
    
    elif mask_type == 'specific':
        # Drop specific patches - return indices of patches to keep
        if specific_patches is None:
            specific_patches = [10, 20, 30]  # Default patches to drop
        
        # Create a list of indices to keep (excluding the ones in specific_patches)
        keep_indices = [i for i in range(total_patches) if i not in specific_patches]
        return torch.tensor(keep_indices).long().to(device)
    
    elif mask_type == 'first_n':
        # Keep only the first N patches (including class token)
        if n_patches is None:
            n_patches = 100  # Default number of patches to keep
        
        # Ensure n_patches doesn't exceed total_patches
        n_patches = min(n_patches, total_patches - 1)
        
        # Return indices 0 to n_patches (including class token at index 0)
        return torch.arange(n_patches + 1).long().to(device)
    
    elif mask_type == 'indices':
        # Directly specify which indices to keep
        if specific_patches is None:
            raise ValueError("For 'indices' mask type, specific_patches must be provided")
        
        # Ensure class token (index 0) is included
        if 0 not in specific_patches:
            specific_patches = [0] + specific_patches
            
        return torch.tensor(specific_patches).long().to(device)
    
    elif mask_type == 'batch_specific':
        # Create different masks for each item in the batch
        # This returns a 2D tensor of shape [batch_size, num_patches_to_keep]
        if specific_patches is None or not isinstance(specific_patches, list) or len(specific_patches) != batch_size:
            raise ValueError("For 'batch_specific', specific_patches must be a list of lists, one per batch item")
        
        # Ensure each batch item's mask includes the class token
        for i in range(batch_size):
            if 0 not in specific_patches[i]:
                specific_patches[i] = [0] + specific_patches[i]
        
        # Find the maximum number of patches in any batch item
        max_patches = max(len(patches) for patches in specific_patches)
        
        # Create a tensor to hold all batch masks
        batch_masks = torch.zeros(batch_size, max_patches).long().to(device)
        
        # Fill in the masks for each batch item
        for i, patches in enumerate(specific_patches):
            batch_masks[i, :len(patches)] = torch.tensor(patches).long()
            
        return batch_masks
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

def preprocess_image(image_path, device=None):
    """
    Load and preprocess an image for ViT inference.
    
    Args:
        image_path (str): Path to the image file
        device (torch.device): Device to load the tensor on
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def run_inference_with_mask(model, input_tensor, patch_mask=None):
    """
    Run inference on an image with optional patch masking.
    
    Args:
        model: The ViT model
        input_tensor (torch.Tensor): Preprocessed input image
        patch_mask (torch.Tensor): Optional patch mask
        
    Returns:
        tuple: (output logits, predicted class)
    """
    with torch.no_grad():
        if patch_mask is None:
            output = model(input_tensor)
        else:
            output = model(input_tensor, patch_mask=patch_mask)
        
        _, pred = output.max(1)
        return output, pred.item()

# Example usage
if __name__ == "__main__":
    # Load model
    weights_path = "/home/shai2403/XAIbench/XAI_Benchmark/SavedModels/vit_timm_standard_mri_ps64_35e.pth"
    model, device = load_vit_model(weights_path)
    
    if model is not None:
        # Preprocess image
        image_path = "/home/shai2403/XAIbench/XAI_Benchmark/output_image.png"
        input_tensor = preprocess_image(image_path, device)
        
        # Create different masks
        all_mask = create_patch_mask('all', device=device)
        specific_mask = create_patch_mask('specific', [10, 20, 30], device=device)
        first_n_mask = create_patch_mask('first_n', n_patches=100, device=device)
        
        # Run inference
        output_normal, pred_normal = run_inference_with_mask(model, input_tensor)
        output_masked, pred_masked = run_inference_with_mask(model, input_tensor, specific_mask)
        
        print(f"Prediction without masking: {pred_normal}")
        print(f"Prediction with masking: {pred_masked}")