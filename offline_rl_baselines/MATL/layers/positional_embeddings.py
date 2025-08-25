import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Transformer-XL style sinusoidal positional embeddings (relative formulation helper).

    - inv_freq: [D/2] inverse frequencies for sin/cos bases.
    - forward(pos_seq, bsz):
        pos_seq: [T] float/long tensor of absolute positions (can include offset)
        bsz: optional batch size to expand over.
      Returns:
        pos_emb: [T, 1, D] if bsz is None, else [T, bsz, D]
      Note: This module returns a [T, 1, D] tensor that is later broadcast
            over batch. It is used with relative attention blocks.
    """
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        # pos_seq supports absolute offset: e.g., arange(offset+T-1, offset-1, -1)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class LearnablePositionalEmbedding(nn.Module):
    """
    Absolute learnable positional embeddings (GPT-like).

    Args:
      max_len: maximum sequence length supported
      d_model: embedding dimension

    forward(positions):
      positions: either
        - LongTensor [T] with explicit absolute indices (supports offsets)
        - int T, meaning range(0..T-1)
      Returns:
        [T, D] absolute positional embeddings.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:  
        """
        Args:
            positions: tensor of position indices [seq_len] or int for seq_len
        Returns:
            positional embeddings [seq_len, d_model]
        """
        # CRITICAL FIX: Support both position tensor and seq_len int
        if isinstance(positions, torch.Tensor):
            # positions is a tensor of indices (supports position_offset)
            return self.weight[positions]
        else:
            # positions is seq_len (backward compatibility)
            seq_len = positions
        return self.weight[:seq_len]


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Absolute sinusoidal embeddings that support a position offset.

    forward(seq_len, position_offset=0):
      seq_len: int sequence length
      position_offset: int offset added to positions (e.g., sliding window start)
      Returns:
        [T, D] tensor, where T = seq_len.

    Implementation details:
    - positions = arange(offset, offset+T)
    - pe[:, 0::2] = sin(positions * div_term)
      pe[:, 1::2] = cos(positions * div_term)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, seq_len: int, position_offset: int = 0, device: torch.device = None) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings
        
        Args:
            seq_len: length of sequence
            position_offset: offset to add to positions
            device: device to create embeddings on
            
        Returns:
            positional embeddings [seq_len, d_model]
        """
        if device is None:
            device = torch.device('cpu')
            
        positions = torch.arange(position_offset, position_offset + seq_len, dtype=torch.float, device=device)
        
        # Create sinusoidal embeddings
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * 
                            -(torch.log(torch.tensor(10000.0, device=device)) / self.d_model))
        
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        # Handle case where d_model is odd
        cos_indices = torch.arange(1, self.d_model, 2, device=device)
        if len(cos_indices) > 0:
            pe[:, cos_indices] = torch.cos(positions.unsqueeze(1) * div_term[:len(cos_indices)])
        
        return pe