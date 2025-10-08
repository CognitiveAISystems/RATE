import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - a simplified alternative to LayerNorm.

    Instead of subtracting the mean and then normalizing like LayerNorm,
    RMSNorm just divides by the root mean square of the inputs. This makes
    it computationally cheaper while often working just as well.

    Args:
        d_model: Size of the last dimension in the input
        eps: Small value to avoid division by zero
        elementwise_affine: Whether to learn a scaling factor for each feature
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
        """Apply RMS normalization to the input tensor.

        Args:
            x: Input tensor with shape (..., d_model)

        Returns:
            Normalized tensor with the same shape as input
        """
        # Calculate the root mean square for normalization
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

        # Normalize by dividing by RMS
        x_normed = x / rms

        # Scale each feature if we have learnable parameters
        if self.elementwise_affine:
            x_normed = x_normed * self.weight
            
        return x_normed
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


def get_norm_layer(norm_type: Optional[str], d_model: int, eps: float = 1e-8) -> nn.Module:
    """Create the right normalization layer based on the specified type.

    Args:
        norm_type: What kind of normalization to use ("layer", "rmsnorm", or None for default)
        d_model: Size of the model dimension
        eps: Small value to help with numerical stability

    Returns:
        The normalization layer you asked for

    Raises:
        ValueError: If you specify an unsupported normalization type
    """
    if norm_type is None or norm_type == "layer":
        return nn.LayerNorm(d_model, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unknown normalization type '{norm_type}'. Try 'layer', 'rmsnorm', or None for default LayerNorm")
