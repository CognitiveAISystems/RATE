import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class PositionalEmbedding(nn.Module):
    """Helper for creating relative positional embeddings used in Transformer-XL style attention.

    This creates sinusoidal position encodings that work with relative attention mechanisms.
    The embeddings help the model understand the relative distance between tokens.
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
    """Trainable position embeddings that learn to represent absolute positions in sequences.

    Unlike fixed sinusoidal embeddings, these are learned parameters that get updated during training.
    Works well for shorter sequences where you want the model to learn position-specific patterns.
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
        # Handle both tensor positions and simple sequence length
        if isinstance(positions, torch.Tensor):
            # positions is a tensor of indices (supports position_offset)
            return self.weight[positions]
        else:
            # positions is seq_len (backward compatibility)
            seq_len = positions
        return self.weight[:seq_len]


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal position embeddings with support for position offsets.

    Creates the classic sinusoidal position encodings that are added to token embeddings.
    The offset parameter is useful for sliding window attention or when processing
    chunks of longer sequences.
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
        
        # Generate the sinusoidal patterns for each position
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * 
                            -(torch.log(torch.tensor(10000.0, device=device)) / self.d_model))
        
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        # Handle odd model dimensions by only using cosine for the last position
        cos_indices = torch.arange(1, self.d_model, 2, device=device)
        if len(cos_indices) > 0:
            pe[:, cos_indices] = torch.cos(positions.unsqueeze(1) * div_term[:len(cos_indices)])
        
        return pe


class RoPEPositionalEmbedding(nn.Module):
    """Rotary Position Embedding - rotates query and key vectors based on their positions.

    Instead of adding position info like traditional embeddings, RoPE rotates the vectors
    by an amount that depends on their absolute positions. This naturally incorporates
    relative position information into the attention computation.

    Works great for longer sequences and doesn't require you to specify a max length upfront.
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Make sure we have an even number of dimensions for the rotation pairs
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
            
        # Calculate how much to rotate based on position
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotation matrices for efficiency
        self._precompute_rotations(max_seq_len)
    
    def _precompute_rotations(self, max_seq_len: int):
        """Precompute cos and sin values for all positions up to max_seq_len"""
        # positions shape: [max_seq_len]
        positions = torch.arange(max_seq_len, dtype=torch.float)
        
        # freqs shape: [max_seq_len, d_model // 2]
        freqs = torch.outer(positions, self.inv_freq)
        
        # emb shape: [max_seq_len, d_model // 2]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # cos_cached, sin_cached shape: [max_seq_len, d_model]
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.
        
        Args:
            x: input tensor [..., dim]
            
        Returns:
            rotated tensor [..., dim]
        """
        dim = x.shape[-1]
        x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            q: query tensor [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, d_model]
            k: key tensor [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, d_model]
            positions: position indices [seq_len] (absolute positions)
            
        Returns:
            tuple of (rotated_q, rotated_k)
        """
        # Handle different input formats
        if q.dim() == 4:  # [batch_size, seq_len, num_heads, head_dim]
            batch_size, seq_len, num_heads, head_dim = q.shape
            # Reshape to [seq_len, batch_size * num_heads, head_dim] for processing
            q_flat = q.transpose(0, 1).reshape(seq_len, batch_size * num_heads, head_dim)
            k_flat = k.transpose(0, 1).reshape(seq_len, batch_size * num_heads, head_dim)
            reshape_back = True
        else:  # [seq_len, batch_size, d_model]
            seq_len, batch_size, d_model = q.shape
            q_flat = q
            k_flat = k
            reshape_back = False
        
        # Check if we need to extend the cache for longer sequences
        max_pos = positions.max().item()
        if max_pos >= self.cos_cached.size(0):
            # Extend the cache to accommodate larger positions
            new_max_seq_len = max(max_pos + 1, self.max_seq_len * 2)
            self._precompute_rotations(new_max_seq_len)
            self.max_seq_len = new_max_seq_len
        
        # Get cos and sin for the given positions
        # positions: [seq_len], cos/sin: [seq_len, d_model]
        cos = self.cos_cached[positions]  # [seq_len, d_model]
        sin = self.sin_cached[positions]  # [seq_len, d_model]
        
        # For head dimension, we need to match the actual dimension being used
        if reshape_back:  # Working with head_dim
            actual_dim = q_flat.shape[-1]  # head_dim
            cos = cos[..., :actual_dim]  # [seq_len, head_dim]
            sin = sin[..., :actual_dim]  # [seq_len, head_dim]
        
        # Expand for batch dimension: [seq_len, 1, actual_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = (q_flat * cos) + (self._rotate_half(q_flat) * sin)
        k_embed = (k_flat * cos) + (self._rotate_half(k_flat) * sin)
        
        # Reshape back if needed
        if reshape_back:
            q_embed = q_embed.reshape(seq_len, batch_size, num_heads, head_dim).transpose(0, 1)
            k_embed = k_embed.reshape(seq_len, batch_size, num_heads, head_dim).transpose(0, 1)
        
        return q_embed, k_embed
    
    def forward(self, seq_len: int, position_offset: int = 0, device: torch.device = None) -> torch.Tensor:
        """
        Generate position indices for RoPE (for compatibility with other position embeddings).
        
        Args:
            seq_len: length of sequence
            position_offset: offset to add to positions
            device: device to create positions on
            
        Returns:
            position indices [seq_len]
        """
        if device is None:
            device = self.inv_freq.device
            
        positions = torch.arange(position_offset, position_offset + seq_len, device=device)
        return positions


class YaRNPositionalEmbedding(nn.Module):
    """YaRN - an improved version of RoPE that handles much longer sequences better.

    YaRN modifies the rotation frequencies in RoPE to work well with sequences much longer
    than what the model was originally trained on. It uses different scaling strategies
    for high-frequency and low-frequency components to maintain good performance across
    different sequence lengths.

    Great for when you want to apply a model trained on shorter sequences to much longer inputs.
    """
    
    def __init__(self,
                 head_dim: int,
                 L_train: int = 2048,
                 L_ext: int = 32768,
                 base: float = 10000.0,
                 m: int = 128,
                 beta: float = 32.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        
        self.head_dim = head_dim
        self.half = head_dim // 2
        self.L_train = L_train
        self.L_ext = L_ext
        self.base = base
        self.m = min(m, self.half)  # Ensure m doesn't exceed available dimensions
        self.beta = beta
        
        # Compute base inverse frequencies for head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, self.half).float() / self.half))
        
        # Apply YaRN scaling
        s = float(L_ext) / float(L_train)
        scales = torch.ones_like(inv_freq)
        
        # Interpolation for first m dimensions (high frequencies)
        if s <= 1.0 + 1e-6:
            scales[:self.m] = 1.0 / max(s, 1e-12)  # Safe division for context reduction
        else:
            scales[:self.m] = 1.0 / s
        
        # Extrapolation for remaining dimensions (low frequencies)
        if self.m < len(scales):
            if s <= 1.0 + 1e-6:
                scales[self.m:] = 1.0  # No scaling if s ≤ 1 (context reduction or no change)
            else:
                import math
                log_beta_s = math.log(s) / math.log(beta)
                scales[self.m:] = 1.0 / log_beta_s
        
        # Register scaled frequencies
        self.register_buffer("inv_freq", inv_freq * scales)
        
        # Initialize cache
        self.max_seq_len = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
    
    def _maybe_grow_cache(self, needed_len: int, device, dtype):
        """Grow cache if needed to accommodate longer sequences"""
        if needed_len <= self.max_seq_len:
            return
        
        # Grow cache to at least needed_len, or double current size
        T = max(needed_len, self.max_seq_len * 2 if self.max_seq_len else needed_len)
        
        # Compute positions and frequencies
        pos = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)  # [T, half]
        
        # Cache cos and sin
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)
        self.max_seq_len = T
    
    @staticmethod
    def _rotate_half(x):
        """Rotate half the hidden dims of the input"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply YaRN rotary position embedding to query and key tensors.
        
        Args:
            q: query tensor [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, d_model]
            k: key tensor [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, d_model]
            positions: position indices [seq_len] (absolute positions)
            
        Returns:
            tuple of (rotated_q, rotated_k)
        """
        # Ensure cache is large enough
        T = int(positions.max().item()) + 1
        self._maybe_grow_cache(T, q.device, q.dtype)
        
        # Get cos and sin for the given positions
        cos = self.cos_cached[positions]  # [T, half]
        sin = self.sin_cached[positions]  # [T, half]
        
        # Handle different input formats
        if q.dim() == 4:  # [batch_size, seq_len, num_heads, head_dim]
            # Get the actual head_dim from tensor
            head_dim = q.shape[-1]
            
            # Duplicate cos/sin to match head_dim
            cos = torch.cat([cos, cos], dim=-1)  # [T, head_dim]
            sin = torch.cat([sin, sin], dim=-1)  # [T, head_dim]
            
            # Slice to actual head_dim if needed
            cos = cos[..., :head_dim]  # [T, head_dim]
            sin = sin[..., :head_dim]  # [T, head_dim]
            
            # Add dimensions for broadcasting: [1, T, 1, head_dim]
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
            
        else:  # [seq_len, batch_size, d_model]
            # For this format, we expect d_model to be head_dim when called from attention
            actual_dim = q.shape[-1]
            
            # Duplicate cos/sin to match actual_dim
            cos = torch.cat([cos, cos], dim=-1)  # [T, actual_dim]
            sin = torch.cat([sin, sin], dim=-1)  # [T, actual_dim]
            
            # Slice to actual dimension
            cos = cos[..., :actual_dim].unsqueeze(1)  # [T, 1, actual_dim]
            sin = sin[..., :actual_dim].unsqueeze(1)  # [T, 1, actual_dim]
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        
        return q_rot, k_rot
    
    def forward(self, seq_len: int, position_offset: int = 0, device: torch.device = None) -> torch.Tensor:
        """
        Generate position indices for YaRN (for compatibility with other position embeddings).
        
        Args:
            seq_len: length of sequence
            position_offset: offset to add to positions
            device: device to create positions on
            
        Returns:
            position indices [seq_len]
        """
        if device is None:
            device = self.inv_freq.device
            
        positions = torch.arange(position_offset, position_offset + seq_len, device=device)
        return positions


class ALiBiPositionalEmbedding(nn.Module):
    """Attention with Linear Biases - adds distance-based penalties to attention scores.

    Instead of using explicit position embeddings, ALiBi adds a penalty to attention scores
    based on how far apart tokens are. Closer tokens get less penalty, distant tokens get more.

    The beauty of this approach is that it has no sequence length limit and adds no parameters -
    you can train on short sequences and apply to arbitrarily long ones without retraining.

    Works especially well for tasks where you need to handle variable-length inputs.
    """
    
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        
        # Compute ALiBi slopes for each attention head
        slopes = self._get_alibi_slopes(n_heads)
        self.register_buffer('slopes', slopes)  # [n_heads]
        
        # Cache for bias matrices
        self.max_cached_len = 0
        self.register_buffer('cached_bias', torch.empty(0), persistent=False)
    
    def _get_alibi_slopes(self, n_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes for attention heads.
        
        The slopes follow the pattern from the original paper:
        - For powers of 2: slopes = [1/2^(i/n) for i in range(n)]
        - For non-powers of 2: combine two power-of-2 sequences
        
        Args:
            n_heads: number of attention heads
            
        Returns:
            slopes tensor [n_heads] (negative values for distance penalty)
        """
        def slopes_power_of_2(n):
            start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if (math.log2(n_heads)).is_integer():
            # n_heads is a power of 2
            slopes = slopes_power_of_2(n_heads)
        else:
            # n_heads is not a power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes = (
                slopes_power_of_2(closest_power_of_2) +
                slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
            )
        
        # keep slopes positive; sign is handled when building the bias
        return torch.tensor(slopes, dtype=torch.float32)

    
    def _build_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build ALiBi bias matrix for self-attention (square matrix).
        
        Args:
            seq_len: sequence length
            device: target device
            dtype: target dtype
            
        Returns:
            bias matrix [1, n_heads, seq_len, seq_len] ready for broadcast with [B, H, T, T] scores
        """
        # Create position indices
        pos = torch.arange(seq_len, device=device)
        
        # distance we penalize: (query index - key index) >= 0 for past keys
        # Shape: [seq_len, seq_len]
        distances = pos[:, None] - pos[None, :] # [T, T]
        
        # Apply slopes to get bias for each head
        # slopes: [n_heads] -> [n_heads, 1, 1]
        # distances: [seq_len, seq_len] -> [1, seq_len, seq_len]
        # Result: [n_heads, seq_len, seq_len]
        slopes = self.slopes.to(device=device, dtype=dtype)  # POSITIVE slopes
        bias = -slopes.view(-1, 1, 1) * distances.view(1, seq_len, seq_len)  # [H, T, T]
        
        # Return [1, n_heads, seq_len, seq_len] for easy broadcast with [B, H, T, T]
        return bias.unsqueeze(0)
    
    def get_bias(self, seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Get ALiBi bias matrix for self-attention.
        
        Args:
            seq_len: sequence length
            device: target device (defaults to slopes device)
            dtype: target dtype (defaults to slopes dtype)
            
        Returns:
            bias matrix [1, n_heads, seq_len, seq_len]
        """
        if device is None:
            device = self.slopes.device
        if dtype is None:
            dtype = self.slopes.dtype
            
        # Check if we need to rebuild cache
        if seq_len > self.max_cached_len or self.cached_bias.numel() == 0:
            # Build new bias matrix
            bias = self._build_bias(seq_len, device, dtype)
            
            # Update cache
            self.register_buffer('cached_bias', bias, persistent=False)
            self.max_cached_len = seq_len
            
            return bias
        else:
            # Use cached bias matrix (slice if needed)
            cached_bias = self.cached_bias.to(device=device, dtype=dtype)
            return cached_bias[:, :, :seq_len, :seq_len]
    
    def get_bias_from_positions(
        self, 
        q_pos: torch.Tensor, 
        k_pos: torch.Tensor,
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Get ALiBi bias matrix for arbitrary query and key positions (e.g., cross-attention).
        
        Args:
            q_pos: [T_q] absolute indices of query positions
            k_pos: [T_k] absolute indices of key positions  
            device: target device (defaults to slopes device)
            dtype: target dtype (defaults to slopes dtype)
            
        Returns:
            bias matrix [1, n_heads, T_q, T_k]
        """
        if device is None:
            device = self.slopes.device
        if dtype is None:
            dtype = self.slopes.dtype
            
        q_pos = q_pos.to(device)
        k_pos = k_pos.to(device)
        
        # penalize (query - key) distance; past keys => positive distance
        # Shape: [T_q, T_k]
        distances = q_pos[:, None] - k_pos[None, :]  # [T_q, T_k]
        
        # Apply slopes to get bias for each head
        # slopes: [n_heads] -> [n_heads, 1, 1]
        # distances: [T_q, T_k] -> [1, T_q, T_k]
        # Result: [n_heads, T_q, T_k]
        slopes = self.slopes.to(device=device, dtype=dtype)  # POSITIVE slopes
        bias = -slopes.view(-1, 1, 1) * distances.view(1, *distances.shape)
        
        # Return [1, n_heads, T_q, T_k] for easy broadcast with [B, H, T_q, T_k]
        return bias.unsqueeze(0)
    
    def forward(self, seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Forward pass to get ALiBi bias matrix for self-attention.
        
        Args:
            seq_len: sequence length
            device: target device
            dtype: target dtype
            
        Returns:
            bias matrix [1, n_heads, seq_len, seq_len]
        """
        return self.get_bias(seq_len, device, dtype)