import torch
import torch.nn as nn
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional causal masking and relative bias."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_out = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        rel_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-head attention with optional causal masking and relative bias.

        Args:
            query: [B, T_q, D]
            key:   [B, T_k, D]
            value: [B, T_v, D]
            mask:  optional boolean mask broadcastable to [B, H, T_q, T_k]
                   (this code assumes it is pre-broadcasted or shaped like [1,1,T_q,T_k])
            is_causal: if True, apply causal masking on top of provided mask
            rel_bias: optional relative bias tensor, either [B, T_q, T_k, H] or [B, H, T_q, T_k]

        Returns:
            output: [B, T_q, D]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query)  # (B, T_q, D)
        K = self.w_k(key)    # (B, T_k, D)
        V = self.w_v(value)  # (B, T_v, D)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_q, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_k, d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_v, d_k)
        

        # Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / math.sqrt(self.d_k)  # (B, H, T_q, T_k)

        # Add relative positional bias if provided
        if rel_bias is not None:
            # rel_bias can be (B, T_q, T_k, H) or (B, H, T_q, T_k)
            if rel_bias.dim() == 4 and rel_bias.shape[-1] == self.n_heads and rel_bias.shape[1] == scores.shape[2]:
                # (B, T_q, T_k, H) -> (B, H, T_q, T_k)
                rel = rel_bias.permute(0, 3, 1, 2)
            else:
                # assume (B, H, T_q, T_k)
                rel = rel_bias
            scores = scores + rel

        # Apply token mask if provided
        if mask is not None:
            # mask is already in [1,1,Q,K] in the calling code
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply causal masking for autoregressive generation
        if is_causal:
            t_q, t_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device), 1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Compute attention weights and apply dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)

        # Apply attention to values
        attention_output = torch.einsum('bhqk,bhkd->bhqd', attn, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        projected = self.w_o(attention_output)
        
        return self.dropout_out(projected)


class YaRNMultiHeadAttention(nn.Module):
    """Multi-head attention with YaRN positional embeddings for improved length extrapolation."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_out = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        rel_bias: Optional[torch.Tensor] = None,
        yarn_embeddings = None,
        yarn_positions = None
    ) -> torch.Tensor:
        """
        Multi-head attention with YaRN support.

        Args:
            query: [B, T_q, D]
            key:   [B, T_k, D]
            value: [B, T_v, D]
            mask:  optional boolean mask broadcastable to [B, H, T_q, T_k]
            is_causal: if True, apply causal masking on top of provided mask
            rel_bias: optional relative bias tensor (ignored when using YaRN)
            yarn_embeddings: YaRN embedding module (YaRNPositionalEmbedding instance)
            yarn_positions: position indices for YaRN [T_q] or [T_k] (assumes same for q and k)

        Returns:
            output: [B, T_q, D]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query)  # (B, T_q, D)
        K = self.w_k(key)    # (B, T_k, D)
        V = self.w_v(value)  # (B, T_v, D)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_q, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_k, d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_v, d_k)
        
        # Apply YaRN if provided
        if yarn_embeddings is not None and yarn_positions is not None:
            # YaRN expects [seq_len, batch_size * num_heads, head_dim] format
            # Convert from [B, H, T, d_k] to [T, B*H, d_k]
            T_q = Q.size(2)
            T_k = K.size(2)
            
            q_yarn = Q.transpose(1, 2).contiguous().view(T_q, batch_size * self.n_heads, self.d_k)  # [T_q, B*H, d_k]
            k_yarn = K.transpose(1, 2).contiguous().view(T_k, batch_size * self.n_heads, self.d_k)  # [T_k, B*H, d_k]
            
            # Apply YaRN rotary position embeddings
            q_yarn, k_yarn = yarn_embeddings.apply_rotary_pos_emb(q_yarn, k_yarn, yarn_positions)
            
            # Convert back to [B, H, T, d_k]
            Q = q_yarn.view(T_q, batch_size, self.n_heads, self.d_k).transpose(0, 1).transpose(1, 2)
            K = k_yarn.view(T_k, batch_size, self.n_heads, self.d_k).transpose(0, 1).transpose(1, 2)

        # Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / math.sqrt(self.d_k)  # (B, H, T_q, T_k)

        # Add relative positional bias if provided (not typically used with YaRN)
        if rel_bias is not None and yarn_embeddings is None:
            # rel_bias can be (B, T_q, T_k, H) or (B, H, T_q, T_k)
            if rel_bias.dim() == 4 and rel_bias.shape[-1] == self.n_heads and rel_bias.shape[1] == scores.shape[2]:
                # (B, T_q, T_k, H) -> (B, H, T_q, T_k)
                rel = rel_bias.permute(0, 3, 1, 2)
            else:
                # assume (B, H, T_q, T_k)
                rel = rel_bias
            scores = scores + rel

        # Apply token mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply causal masking for autoregressive generation
        if is_causal:
            t_q, t_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device), 1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Compute attention weights and apply dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)

        # Apply attention to values
        attention_output = torch.einsum('bhqk,bhkd->bhqd', attn, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        projected = self.w_o(attention_output)
        
        return self.dropout_out(projected)


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE positional embeddings for better sequence understanding."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_out = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        rel_bias: Optional[torch.Tensor] = None,
        rope_embeddings = None,
        rope_positions = None
    ) -> torch.Tensor:
        """
        Multi-head attention with RoPE support.

        Args:
            query: [B, T_q, D]
            key:   [B, T_k, D]
            value: [B, T_v, D]
            mask:  optional boolean mask broadcastable to [B, H, T_q, T_k]
            is_causal: if True, apply causal masking on top of provided mask
            rel_bias: optional relative bias tensor (ignored when using RoPE)
            rope_embeddings: RoPE embedding module (RoPEPositionalEmbedding instance)
            rope_positions: position indices for RoPE [T_q] or [T_k] (assumes same for q and k)

        Returns:
            output: [B, T_q, D]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query)  # (B, T_q, D)
        K = self.w_k(key)    # (B, T_k, D)
        V = self.w_v(value)  # (B, T_v, D)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_q, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_k, d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_v, d_k)
        
        # Apply RoPE if provided
        if rope_embeddings is not None and rope_positions is not None:
            # RoPE expects [seq_len, batch_size * num_heads, head_dim] format
            # Convert from [B, H, T, d_k] to [T, B*H, d_k]
            T_q = Q.size(2)
            T_k = K.size(2)
            
            q_rope = Q.transpose(1, 2).contiguous().view(T_q, batch_size * self.n_heads, self.d_k)  # [T_q, B*H, d_k]
            k_rope = K.transpose(1, 2).contiguous().view(T_k, batch_size * self.n_heads, self.d_k)  # [T_k, B*H, d_k]
            
            # Apply rotary position embeddings
            q_rope, k_rope = rope_embeddings.apply_rotary_pos_emb(q_rope, k_rope, rope_positions)
            
            # Convert back to [B, H, T, d_k]
            Q = q_rope.view(T_q, batch_size, self.n_heads, self.d_k).transpose(0, 1).transpose(1, 2)
            K = k_rope.view(T_k, batch_size, self.n_heads, self.d_k).transpose(0, 1).transpose(1, 2)

        # Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / math.sqrt(self.d_k)  # (B, H, T_q, T_k)

        # Add relative positional bias if provided (not typically used with RoPE)
        if rel_bias is not None and rope_embeddings is None:
            # rel_bias can be (B, T_q, T_k, H) or (B, H, T_q, T_k)
            if rel_bias.dim() == 4 and rel_bias.shape[-1] == self.n_heads and rel_bias.shape[1] == scores.shape[2]:
                # (B, T_q, T_k, H) -> (B, H, T_q, T_k)
                rel = rel_bias.permute(0, 3, 1, 2)
            else:
                # assume (B, H, T_q, T_k)
                rel = rel_bias
            scores = scores + rel

        # Apply token mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply causal masking for autoregressive generation
        if is_causal:
            t_q, t_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device), 1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Compute attention weights and apply dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)

        # Apply attention to values
        attention_output = torch.einsum('bhqk,bhkd->bhqd', attn, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        projected = self.w_o(attention_output)
        
        return self.dropout_out(projected)


class YaRNMultiHeadAttention(nn.Module):
    """Multi-head attention with YaRN positional embeddings for improved length extrapolation."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_out = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        rel_bias: Optional[torch.Tensor] = None,
        yarn_embeddings = None,
        yarn_positions = None
    ) -> torch.Tensor:
        """
        Multi-head attention with YaRN support.

        Args:
            query: [B, T_q, D]
            key:   [B, T_k, D]
            value: [B, T_v, D]
            mask:  optional boolean mask broadcastable to [B, H, T_q, T_k]
            is_causal: if True, apply causal masking on top of provided mask
            rel_bias: optional relative bias tensor (ignored when using YaRN)
            yarn_embeddings: YaRN embedding module (YaRNPositionalEmbedding instance)
            yarn_positions: position indices for YaRN [T_q] or [T_k] (assumes same for q and k)

        Returns:
            output: [B, T_q, D]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query)  # (B, T_q, D)
        K = self.w_k(key)    # (B, T_k, D)
        V = self.w_v(value)  # (B, T_v, D)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_q, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_k, d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_v, d_k)
        
        # Apply YaRN if provided
        if yarn_embeddings is not None and yarn_positions is not None:
            # YaRN expects [seq_len, batch_size * num_heads, head_dim] format
            # Convert from [B, H, T, d_k] to [T, B*H, d_k]
            T_q = Q.size(2)
            T_k = K.size(2)
            
            q_yarn = Q.transpose(1, 2).contiguous().view(T_q, batch_size * self.n_heads, self.d_k)  # [T_q, B*H, d_k]
            k_yarn = K.transpose(1, 2).contiguous().view(T_k, batch_size * self.n_heads, self.d_k)  # [T_k, B*H, d_k]
            
            # Apply YaRN rotary position embeddings
            q_yarn, k_yarn = yarn_embeddings.apply_rotary_pos_emb(q_yarn, k_yarn, yarn_positions)
            
            # Convert back to [B, H, T, d_k]
            Q = q_yarn.view(T_q, batch_size, self.n_heads, self.d_k).transpose(0, 1).transpose(1, 2)
            K = k_yarn.view(T_k, batch_size, self.n_heads, self.d_k).transpose(0, 1).transpose(1, 2)

        # Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / math.sqrt(self.d_k)  # (B, H, T_q, T_k)

        # Add relative positional bias if provided (not typically used with YaRN)
        if rel_bias is not None and yarn_embeddings is None:
            # rel_bias can be (B, T_q, T_k, H) or (B, H, T_q, T_k)
            if rel_bias.dim() == 4 and rel_bias.shape[-1] == self.n_heads and rel_bias.shape[1] == scores.shape[2]:
                # (B, T_q, T_k, H) -> (B, H, T_q, T_k)
                rel = rel_bias.permute(0, 3, 1, 2)
            else:
                # assume (B, H, T_q, T_k)
                rel = rel_bias
            scores = scores + rel

        # Apply token mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply causal masking for autoregressive generation
        if is_causal:
            t_q, t_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device), 1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Compute attention weights and apply dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)

        # Apply attention to values
        attention_output = torch.einsum('bhqk,bhkd->bhqd', attn, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        projected = self.w_o(attention_output)
        
        return self.dropout_out(projected)


class ALiBiMultiHeadAttention(nn.Module):
    """Multi-head attention with ALiBi positional biases for efficient long-range modeling."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_out = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        rel_bias: Optional[torch.Tensor] = None,
        alibi_embeddings = None,
        alibi_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-head attention with ALiBi support.

        Args:
            query: [B, T_q, D]
            key:   [B, T_k, D]
            value: [B, T_v, D]
            mask:  optional boolean mask broadcastable to [B, H, T_q, T_k]
            is_causal: if True, apply causal masking on top of provided mask
            rel_bias: optional relative bias tensor (ignored when using ALiBi)
            alibi_embeddings: ALiBi embedding module (ALiBiPositionalEmbedding instance)
            alibi_bias: precomputed ALiBi bias matrix [n_heads, seq_len, seq_len]

        Returns:
            output: [B, T_q, D]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query)  # (B, T_q, D)
        K = self.w_k(key)    # (B, T_k, D)
        V = self.w_v(value)  # (B, T_v, D)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_q, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_k, d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_v, d_k)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T_q, T_k)

        # Add ALiBi positional bias if provided
        if alibi_bias is not None:
            # expect shape [1, H, T_q, T_k] – add it directly
            scores = scores + alibi_bias
        elif alibi_embeddings is not None:
            if query.size(1) == key.size(1):
                # self-attention case
                alibi_bias = alibi_embeddings.get_bias(scores.size(-1), scores.device, scores.dtype)  # [1,H,T,T]
            else:
                # fallback for different query/key lengths
                alibi_bias = alibi_embeddings.get_bias(key.size(1), scores.device, scores.dtype)
            scores = scores + alibi_bias

        # Apply padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply causal mask for autoregressive generation
        if is_causal:
            t_q, t_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device), 1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Compute attention weights and apply dropout
        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, T_q, T_k)
        attn_weights = self.dropout_attn(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T_q, d_k)

        # Combine heads and project to output dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (B, T_q, D)
        
        return self.dropout_out(self.w_o(attn_output))