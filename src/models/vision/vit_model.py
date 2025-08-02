"""Vision Transformer (ViT) model implementations."""

import math
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        """Initialize patch embedding.
        
        Args:
            img_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Patch embeddings of shape (batch_size, n_patches, embed_dim)
        """
        batch_size = x.shape[0]
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        """Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate queries, keys, values
        q = self.query(x)  # (batch_size, seq_len, embed_dim)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final projection
        output = self.projection(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        """Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout probability
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(BaseModel):
    """Vision Transformer (ViT) model."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        model_name: str = "vision_transformer",
        device: Optional[torch.device] = None
    ):
        """Initialize Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            model_name: Name of the model
            device: Device to run model on
        """
        super().__init__(num_classes, model_name, device)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other parameters
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification head (use class token)
        cls_token_final = x[:, 0]  # (batch_size, embed_dim)
        logits = self.head(cls_token_final)  # (batch_size, num_classes)
        
        return logits


class ViTWithPatchDrop(VisionTransformer):
    """Vision Transformer with patch dropout capability for ablation studies."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        patch_dropout: float = 0.0,
        model_name: str = "vit_patch_drop",
        device: Optional[torch.device] = None
    ):
        """Initialize ViT with patch dropout.
        
        Args:
            patch_dropout: Probability of dropping patches during training
            Other args same as VisionTransformer
        """
        super().__init__(
            img_size, patch_size, in_channels, num_classes, embed_dim,
            depth, num_heads, mlp_ratio, dropout, model_name, device
        )
        self.patch_dropout = patch_dropout
    
    def drop_patches(
        self, 
        x: torch.Tensor, 
        patch_drop_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drop random patches from input.
        
        Args:
            x: Input tensor with patches (batch_size, n_patches + 1, embed_dim)
            patch_drop_ratio: Ratio of patches to drop (overrides self.patch_dropout)
            
        Returns:
            Tuple of (tensor with dropped patches, mask indicating kept patches)
        """
        if patch_drop_ratio is None:
            patch_drop_ratio = self.patch_dropout
        
        if patch_drop_ratio == 0.0 or not self.training:
            # No dropout, return original tensor
            batch_size, seq_len, _ = x.shape
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            return x, mask
        
        batch_size, seq_len, embed_dim = x.shape
        # seq_len = n_patches + 1 (class token)
        n_patches = seq_len - 1
        
        # Generate random mask (keep class token, drop patches)
        keep_ratio = 1.0 - patch_drop_ratio
        n_keep = max(1, int(n_patches * keep_ratio))
        
        # Create mask for each sample in batch
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        mask[:, 0] = True  # Always keep class token
        
        for i in range(batch_size):
            # Randomly select patches to keep
            patch_indices = torch.randperm(n_patches, device=x.device)[:n_keep] + 1
            mask[i, patch_indices] = True
        
        # Apply mask
        x_dropped = torch.zeros(batch_size, n_keep + 1, embed_dim, device=x.device)
        for i in range(batch_size):
            x_dropped[i] = x[i, mask[i]]
        
        return x_dropped, mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        patch_drop_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """Forward pass with patch dropout.
        
        Args:
            x: Input tensor
            patch_drop_ratio: Patch dropout ratio (overrides default)
            
        Returns:
            Output logits
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply patch dropout
        x, patch_mask = self.drop_patches(x, patch_drop_ratio)
        
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits


def create_vit_small(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """Create a small ViT model."""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        model_name="vit_small",
        **kwargs
    )


def create_vit_base(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """Create a base ViT model.""" 
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        model_name="vit_base",
        **kwargs
    )


def create_vit_large(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """Create a large ViT model."""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
        model_name="vit_large",
        **kwargs
    )


# Registry of ViT variants
VIT_MODELS = {
    "vit_small": create_vit_small,
    "vit_base": create_vit_base,
    "vit_large": create_vit_large,
}


def get_vit_model(model_name: str, num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """Get a ViT model by name.
    
    Args:
        model_name: Name of the ViT variant
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        VisionTransformer instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in VIT_MODELS:
        raise ValueError(
            f"Unknown ViT model: {model_name}. "
            f"Available models: {list(VIT_MODELS.keys())}"
        )
    
    return VIT_MODELS[model_name](num_classes=num_classes, **kwargs)