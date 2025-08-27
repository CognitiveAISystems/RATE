import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    RMSNorm is a simpler alternative to LayerNorm that normalizes
    the inputs using only the root mean square (RMS) statistic,
    without centering (subtracting the mean).
    
    Args:
        d_model: The size of the last dimension of the input tensor
        eps: A value added to the denominator for numerical stability
        elementwise_affine: If True, applies a learnable affine transformation
    """
    
    def __init__(self, d_model: int, eps: float = 1e-8, elementwise_affine: bool = True):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS
        x_normed = x / rms
        
        # Apply learnable scaling if enabled
        if self.elementwise_affine:
            x_normed = x_normed * self.weight
            
        return x_normed
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


def get_norm_layer(norm_type: Optional[str], d_model: int, eps: float = 1e-8) -> nn.Module:
    """Factory function to create normalization layers.
    
    Args:
        norm_type: Type of normalization ("layer", "rmsnorm", or None)
        d_model: Model dimension
        eps: Epsilon for numerical stability
        
    Returns:
        Normalization layer instance
        
    Raises:
        ValueError: If norm_type is not supported
    """
    if norm_type is None or norm_type == "layer":
        return nn.LayerNorm(d_model, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}. Supported types: 'layer', 'rmsnorm', None")
