import torch
import torch.nn as nn
from .normalization import get_norm_layer

class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, pre_lnorm: bool = True, norm_type=None):
        super().__init__()
        self.pre_lnorm = pre_lnorm
        
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.layer_norm = get_norm_layer(norm_type, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(x))

            # residual connection
            output = core_out + x
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(x)

            # residual connection + layer normalization
            output = self.layer_norm(x + core_out)

        return output