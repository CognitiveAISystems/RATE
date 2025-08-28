import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List

from offline_rl_baselines.MATL.layers import (
    RelPartialLearnableMultiHeadAttn, 
    MultiHeadAttention, 
    RoPEMultiHeadAttention,
    YaRNMultiHeadAttention,
    ALiBiMultiHeadAttention,
    FeedForwardNetwork,
    PositionalEmbedding,
    LearnablePositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RoPEPositionalEmbedding,
    YaRNPositionalEmbedding,
    ALiBiPositionalEmbedding,
    MemoryState, RelativeBias,
    RMSNorm, get_norm_layer,
    MoEFeedForwardNetwork
)
from RATE.env_encoders import ObsEncoder, ActEncoder, RTGEncoder, ActDecoder

    
class MATLLayer(nn.Module):
    """
    Memory-Augmented Transformer Layer (MATL)

    High-level flow per layer:
    - Token track (current window only):
      1) Self-attention over tokens in the window (no concatenation with memory)
      2) Cross-attention: tokens read from memory (with relative bias)
      3) Position-wise FFN

    - Memory track (persistent across windows):
      1) Cross-attention: memory writes from tokens (with relative bias, opposite direction)
      2) Position-wise FFN to form candidate memory update
      3) Write policy per batch element:
         - If there are empty slots: write into the first empty slot
         - Else (all filled): write into the least-recently-updated (LRU) slot
         - For LRU writes, apply convex blend: new = alpha * candidate + (1-alpha) * old,
           where alpha = config.lru_blend_alpha (default ~0.999); for empty slots alpha=1.0

    Shapes (seq-first in this module):
      - Tokens h: [T, B, D]
      - Memory vec/pos: vec [B, M, D], pos [B, M] (timestamps/anchors)
      - Relative bias read: [B, T, M, H]; write: [B, H, M, T]
    """
    
    def __init__(self, 
        d_model,
        d_ff,
        n_head, 
        memory_size,
        pos_encoding,
        dropout,
        dropatt,
        pre_lnorm,
        max_seq_len=1000,
        memory_init_std=0.02,
        use_lru=True,
        lru_blend_alpha=0.999,
        memory_dropout=None,
        norm_type=None,
        # MoE parameters
        use_moe=False,
        num_experts=8,
        top_k=2,
        expert_dropout=None,
        load_balancing_loss_coef=0.01,
        use_swiglu=True,
        # DeepSeek-style additions
        use_shared_expert=True,
        n_shared_experts=1,
        shared_d_ff=None,
        routed_d_ff=None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = n_head
        self.memory_size = memory_size
        self.pos_encoding = pos_encoding
        self.pre_lnorm = pre_lnorm
        self.memory_init_std = memory_init_std
        self.use_lru = use_lru
        self.lru_blend_alpha = lru_blend_alpha
        self.memory_dropout = memory_dropout if memory_dropout is not None else dropout
        self.norm_type = norm_type
        # MoE parameters
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = expert_dropout if expert_dropout is not None else dropout
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.use_swiglu = use_swiglu
        # DeepSeekMoE (shared expert) additions
        self.use_shared_expert = use_shared_expert
        self.n_shared_experts = n_shared_experts
        self.shared_d_ff = shared_d_ff
        self.routed_d_ff = routed_d_ff
        
        # --- Use appropriate attention for the pos_encoding type ---
        # The self-attention mechanism must be chosen based on the positional encoding strategy.
        # Relative-position attention has a different signature than standard attention.
        if pos_encoding == 'relative':
            self.self_attention = RelPartialLearnableMultiHeadAttn(
                self.d_model, self.num_heads, dropout, 
                dropatt=dropatt, pre_lnorm=pre_lnorm, norm_type=self.norm_type
            )
        elif pos_encoding == 'rope':
            # For RoPE, use RoPE-compatible MultiHeadAttention
            self.self_attention = RoPEMultiHeadAttention(
                self.d_model, self.num_heads, dropout=dropout, attn_dropout=dropatt
            )
            # Add layer norm and residual connection for the self-attention block
            self.self_attn_norm = get_norm_layer(self.norm_type, self.d_model)
        elif pos_encoding == 'yarn':
            # For YaRN, use YaRN-compatible MultiHeadAttention
            self.self_attention = YaRNMultiHeadAttention(
                self.d_model, self.num_heads, dropout=dropout, attn_dropout=dropatt
            )
            # Add layer norm and residual connection for the self-attention block
            self.self_attn_norm = get_norm_layer(self.norm_type, self.d_model)
        elif pos_encoding == 'alibi':
            # For ALiBi, use ALiBi-compatible MultiHeadAttention
            self.self_attention = ALiBiMultiHeadAttention(
                self.d_model, self.num_heads, dropout=dropout, attn_dropout=dropatt
            )
            # Add layer norm and residual connection for the self-attention block
            self.self_attn_norm = get_norm_layer(self.norm_type, self.d_model)
        else:
            # For 'sinusoidal' or 'learnable' embeddings, use standard MultiHeadAttention
            self.self_attention = MultiHeadAttention(
                self.d_model, self.num_heads, dropout=dropout, attn_dropout=dropatt
            )
            # Add layer norm and residual connection for the self-attention block
            # when not using the built-in ones from RelPartialLearnableMultiHeadAttn
            self.self_attn_norm = get_norm_layer(self.norm_type, self.d_model)

        # Multi-head attention modules for memory interaction are always standard
        # Cross-attention 1: tokens read from memory
        self.cross_attention1 = MultiHeadAttention(
            self.d_model, self.num_heads, dropout=dropout, attn_dropout=dropatt
        )
        # Cross-attention 2: memory writes from tokens
        self.cross_attention2 = MultiHeadAttention(
            self.d_model, self.num_heads, dropout=dropout, attn_dropout=dropatt
        )
        
        # Relative bias for cross-attention
        self.cross_rel_bias = RelativeBias(self.num_heads, max_seq_len)
        
        # Feed-forward networks - use MoE if enabled
        self.token_ffn = MoEFeedForwardNetwork(
            d_model=self.d_model, d_ff=d_ff, dropout=dropout, pre_lnorm=pre_lnorm,
            norm_type=self.norm_type,
            use_moe=self.use_moe,
            num_experts=self.num_experts,
            top_k=self.top_k,                              # TOTAL K
            expert_dropout=self.expert_dropout,
            load_balancing_loss_coef=self.load_balancing_loss_coef,
            use_swiglu=self.use_swiglu,
            use_shared_expert=self.use_shared_expert,
            n_shared_experts=self.n_shared_experts,
            shared_d_ff=self.shared_d_ff or d_ff,         # default shared wide = d_ff
            routed_d_ff=self.routed_d_ff or d_ff          # default routed = d_ff
        )
        self.memory_ffn = MoEFeedForwardNetwork(
            d_model=self.d_model, d_ff=d_ff, dropout=dropout, pre_lnorm=pre_lnorm,
            norm_type=self.norm_type,
            use_moe=self.use_moe,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_dropout=self.expert_dropout,
            load_balancing_loss_coef=self.load_balancing_loss_coef,
            use_swiglu=self.use_swiglu,
            use_shared_expert=self.use_shared_expert,
            n_shared_experts=self.n_shared_experts,
            shared_d_ff=self.shared_d_ff or d_ff,
            routed_d_ff=self.routed_d_ff or d_ff
        )

        
        # Layer normalization
        self.token_norm_cross = get_norm_layer(self.norm_type, self.d_model)
        self.memory_norm_cross = get_norm_layer(self.norm_type, self.d_model)
        
        # Initialize parameters
        self._init_parameters()

    def _apply_sublayer(self, x: torch.Tensor, norm: nn.Module, fn, pre_lnorm: bool) -> torch.Tensor:
        """Apply a sublayer with consistent Pre-LN/Post-LN behavior.

        If pre_lnorm is True: y = x + fn(norm(x))
        Else:                 y = norm(x + fn(x))
        """
        if pre_lnorm:
            return x + fn(norm(x))
        else:
            return norm(x + fn(x))
    
    def _init_parameters(self):
        for module in [
            self.self_attention, 
            self.cross_attention1, 
            self.cross_attention2, 
            self.token_ffn, 
            self.memory_ffn
        ]:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                # LayerNorm and RMSNorm parameters are left with their default initialization (weight=1, bias=0 for LayerNorm)
    

    def init_memory(self, batch_size: int, device: torch.device) -> MemoryState:
        # Use model parameter dtype for memory tensors to respect selected precision
        param_dtype = next(self.parameters()).dtype
        vec = torch.randn(
            batch_size, self.memory_size, self.d_model, device=device, dtype=param_dtype
        )
        vec = vec * self.memory_init_std
        # initialize all slots as “unwritten”
        pos = torch.full((batch_size, self.memory_size), -1, dtype=torch.long, device=device)
        return MemoryState(vec, pos)
    
    def forward(
        self,
        h: torch.Tensor,                    # [T, B, D]
        memory_state: MemoryState,          # .vec: [B, M, D], .pos: [B, M]
        tok_pos: torch.LongTensor,          # [T] absolute positions of tokens
        pos_emb: Optional[torch.Tensor],    # [T, 1, D]
        r_w_bias: Optional[torch.Tensor],
        r_r_bias: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        pos_embeddings = None  # RoPE/YaRN embeddings
    ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Forward pass of a single MATL layer.

        Args:
            h: [T, B, D] token hidden states for the current window (seq-first)
            memory_state: MemoryState(vec [B, M, D], pos [B, M]) from the previous window
            tok_pos: [T] absolute token positions (newest-first expected by caller)
            pos_emb: [T, 1, D] relative position embeddings (used only if pos_encoding='relative')
            r_w_bias: [H, d_head] content bias for relative attention (relative mode only)
            r_r_bias: [H, d_head] position bias for relative attention (relative mode only)
            mask: optional causal mask for tokens (current window only)

        Returns:
            output: [T, B, D] updated token states for this layer
            memory: MemoryState(vec [B, M, D], pos [B, M]) updated according to write policy
            aux_loss: Optional auxiliary loss from MoE (load balancing loss)
        """

        mem_vec, mem_pos = memory_state.vec, memory_state.pos
        T, B, D = h.size(0), h.size(1), h.size(2)

        # ========== Token Track ==========
        
        # --- Self-attention on current tokens only (MATL design) ---
        if self.pos_encoding == 'relative':
            # CRITICAL FIX: For relative attention, pass None as mems since MATL doesn't use memory concatenation
            # Memory interacts through separate cross-attention mechanisms, not through mems parameter
            h = self.self_attention(h, pos_emb, r_w_bias, r_r_bias, attn_mask=mask, mems=None)
        elif self.pos_encoding == 'rope':
            # For RoPE, use RoPE-compatible attention
            # Prepare mask format for RoPEMultiHeadAttention
            attn_mask = None
            if mask is not None:
                attn_mask = mask.squeeze(-1) if len(mask.shape) == 3 else mask

            def rope_self_attn_fn(x_tokens: torch.Tensor) -> torch.Tensor:
                attn_out_local = self.self_attention(
                    x_tokens.transpose(0, 1),  # [B, T, D]
                    x_tokens.transpose(0, 1),
                    x_tokens.transpose(0, 1),
                    mask=attn_mask,
                    is_causal=True,
                    rope_embeddings=pos_embeddings,
                    rope_positions=tok_pos  # Use absolute positions for RoPE
                )
                return attn_out_local.transpose(0, 1)

            h = self._apply_sublayer(h, self.self_attn_norm, rope_self_attn_fn, self.pre_lnorm)
        elif self.pos_encoding == 'yarn':
            # For YaRN, use YaRN-compatible attention
            # Prepare mask format for YaRNMultiHeadAttention
            attn_mask = None
            if mask is not None:
                attn_mask = mask.squeeze(-1) if len(mask.shape) == 3 else mask

            def yarn_self_attn_fn(x_tokens: torch.Tensor) -> torch.Tensor:
                attn_out_local = self.self_attention(
                    x_tokens.transpose(0, 1),  # [B, T, D]
                    x_tokens.transpose(0, 1),
                    x_tokens.transpose(0, 1),
                    mask=attn_mask,
                    is_causal=True,
                    yarn_embeddings=pos_embeddings,  # YaRN embeddings
                    yarn_positions=tok_pos  # Use absolute positions for YaRN
                )
                return attn_out_local.transpose(0, 1)

            h = self._apply_sublayer(h, self.self_attn_norm, yarn_self_attn_fn, self.pre_lnorm)
        elif self.pos_encoding == 'alibi':
            # For ALiBi, use ALiBi-compatible attention
            # Prepare mask format for ALiBiMultiHeadAttention
            attn_mask = None
            if mask is not None:
                attn_mask = mask.squeeze(-1) if len(mask.shape) == 3 else mask

            def alibi_self_attn_fn(x_tokens: torch.Tensor) -> torch.Tensor:
                attn_out_local = self.self_attention(
                    x_tokens.transpose(0, 1),  # [B, T, D]
                    x_tokens.transpose(0, 1),
                    x_tokens.transpose(0, 1),
                    mask=attn_mask,
                    is_causal=True,
                    alibi_embeddings=pos_embeddings  # ALiBi embeddings
                )
                return attn_out_local.transpose(0, 1)

            h = self._apply_sublayer(h, self.self_attn_norm, alibi_self_attn_fn, self.pre_lnorm)
        else:
            # For non-relative modes, apply consistent pre/post-LN around self-attention
            # Prepare mask format for MultiHeadAttention
            attn_mask = None
            if mask is not None:
                attn_mask = mask.squeeze(-1) if len(mask.shape) == 3 else mask

            def self_attn_fn(x_tokens: torch.Tensor) -> torch.Tensor:
                attn_out_local = self.self_attention(
                    x_tokens.transpose(0, 1),  # [B, T, D]
                    x_tokens.transpose(0, 1),
                    x_tokens.transpose(0, 1),
                    mask=attn_mask,
                    is_causal=True
                )
                return attn_out_local.transpose(0, 1)

            h = self._apply_sublayer(h, self.self_attn_norm, self_attn_fn, self.pre_lnorm)

        # Memory fusion — tokens learn from memory via cross-attention (consistent LN)
        # --- compute rel-bias for read ---
        # tok_pos: [T], mem_pos: [B, M]
        # rel_read encodes per-head bias for distances Δ = tok_pos − mem_pos
        rel_read = self.cross_rel_bias(tok_pos, mem_pos)    # [B, T, M, H]

        def token_reads_from_memory(x_tokens: torch.Tensor) -> torch.Tensor:
            r_local = self.cross_attention1(
                x_tokens.transpose(0, 1),  # Q: [B, T, D]
                mem_vec,                   # K,V: [B, M, D]
                mem_vec,
                is_causal=False,
                rel_bias=rel_read
            )
            return r_local.transpose(0, 1)

        h = self._apply_sublayer(h, self.token_norm_cross, token_reads_from_memory, self.pre_lnorm)
        
        # Token FFN (handles its own norm and residual)
        h, token_aux_loss = self.token_ffn(h)
        
        # ========== Memory Track ==========
        mem_res = mem_vec
        # # Memory cross-attention — memory learns from tokens (consistent LN)
        # # --- compute rel-bias for write (memory as Q, tokens as K) ---
        # # For mem→tok we need distances (mem_pos − tok_pos) = −(tok_pos − mem_pos).
        # # Using rel_read (tok→mem), we permute and negate to obtain [B, H, M, T].
        # rel_write = - rel_read.permute(0, 3, 2, 1)  # [B, H, M, T]

        # def memory_writes_from_tokens(m_memory: torch.Tensor) -> torch.Tensor:
        #     u_attn_local = self.cross_attention2(
        #         m_memory,                  # Q: [B, M, D]
        #         h.transpose(0, 1),  # K: [B, T, D]
        #         h.transpose(0, 1),  # V: [B, T, D]
        #         is_causal=False,
        #         rel_bias=rel_write
        #     )
        #     return u_attn_local

        # Memory cross-attention — memory learns from tokens (consistent LN)
        # --- compute rel-bias for write (memory as Q, tokens as K) ---
        # For mem→tok we need distances (mem_pos − tok_pos).
        rel_write = self.cross_rel_bias.mem_to_tok(mem_pos, tok_pos)  # [B, H, M, T]

        def memory_writes_from_tokens(m_memory: torch.Tensor) -> torch.Tensor:
            u_attn_local = self.cross_attention2(
                m_memory,                  # Q: [B, M, D]
                h.transpose(0, 1),        # K: [B, T, D]
                h.transpose(0, 1),        # V: [B, T, D]
                is_causal=False,
                rel_bias=rel_write
            )
            return u_attn_local

        u = self._apply_sublayer(mem_vec, self.memory_norm_cross, memory_writes_from_tokens, self.pre_lnorm)

        u_processed, memory_aux_loss = self.memory_ffn(u)
        # Apply dropout to memory updates for regularization on long sequences
        if self.training:
            u_processed = F.dropout(u_processed, p=self.memory_dropout, training=True)

        
        # 1) Define empty slots
        empty = mem_pos < 0                                  # [B, M]
        first_empty = (empty.float().cumsum(-1) == 1)        # one-hot for the first empty

        if self.use_lru:
            # print(mem_pos[0])
            # 3) If memory is fully occupied → select LRU
            all_filled = (~empty).all(dim=-1, keepdim=True)      # [B,1]
            _, lru_idx = mem_pos.min(dim=-1)                     # index of the minimum mem_pos
            lru_one_hot = torch.zeros_like(mem_pos, dtype=torch.bool)
            lru_one_hot.scatter_(1, lru_idx.unsqueeze(1), True)  # one-hot for the LRU slot

            # 4) Write mask: if all occupied → write to LRU, otherwise → to the first empty
            write_mask = torch.where(
                all_filled.expand_as(first_empty),
                lru_one_hot,
                first_empty
            )

            # 5) Update vectors and positions
            # For LRU we use a convex combination: alpha * new + (1-alpha) * old
            # For writing to an empty slot we use a hard replacement (alpha=1.0)
            alpha_base = torch.ones_like(mem_res[..., :1])
            lru_alpha = torch.full_like(alpha_base, self.lru_blend_alpha)
            alpha = torch.where(lru_one_hot.unsqueeze(-1), lru_alpha, alpha_base)
            blended = alpha * u_processed + (1.0 - alpha) * mem_res
            new_vec = torch.where(write_mask.unsqueeze(-1), blended, mem_res)
            anchor = tok_pos[0].view(1, 1).expand_as(mem_pos)
            new_pos = torch.where(write_mask, anchor, mem_pos)

            # new_vec - updated memory, where new slots are replaced, others are kept
            # new_pos - updated positions of slots, where new slots are replaced, others are kept
        else:
            # Simple replacement policy — only the first empty slot
            write_mask = first_empty
            new_vec = torch.where(write_mask.unsqueeze(-1), u_processed, mem_res)
            anchor = tok_pos[0].view(1, 1).expand_as(mem_pos)
            new_pos = torch.where(write_mask, anchor, mem_pos)
        
        new_memory_state = MemoryState(new_vec, new_pos)
        
        # Collect auxiliary losses from MoE if any
        aux_losses = []
        if token_aux_loss is not None:
            aux_losses.append(token_aux_loss)
        if memory_aux_loss is not None:
            aux_losses.append(memory_aux_loss)
        
        total_aux_loss = sum(aux_losses) if aux_losses else None
        
        # Return hidden representations (vocabulary projection done by MATLModel)
        return h, new_memory_state, total_aux_loss


class MATLModel(nn.Module):
    def __init__(self,
        state_dim, 
        act_dim,
        d_model=64, 
        d_ff=512,
        n_layer=4,
        n_head=4,
        max_seq_len=1000,
        dropout=0.1,
        dropatt=0.05,
        pre_lnorm=True,
        memory_size=16,
        env_name='mujoco',
        # MATL-specific parameters
        memory_init_std=0.02,
        detach_memory=True,
        use_causal_self_attn_mask=True,
        use_lru=True,
        lru_blend_alpha=0.999,
        pos_type="relative",  # Will use pos_encoding if None
        train_stride=None,  # Will use context_length if None
        padding_idx=None,
        memory_dropout=None,  # Additional dropout for memory updates
        dtype="float32",  # Model dtype: "float32", "float64", "bfloat16"
        sequence_format="sra",  # Sequence format: "s", "sa", "sra", "sr"
        norm_type=None,  # Normalization type: "layer", "rmsnorm", or None (defaults to LayerNorm)
        # MoE parameters
        use_moe=False,  # Whether to use Mixture of Experts
        num_experts=8,  # Number of experts in MoE
        top_k=2,  # Number of experts to select per token
        expert_dropout=None,  # Dropout for experts (uses model dropout if None)
        load_balancing_loss_coef=0.01,  # Coefficient for load balancing loss
        use_swiglu=True,  # Whether to use SwiGLU activation in FFN/experts
        # DeepSeek-style MoE additions
        use_shared_expert=True,
        n_shared_experts=1,
        shared_d_ff=None,
        routed_d_ff=None,
        **kwargs
    ):
        super().__init__()
        
        # Core model parameters
        self.d_model = d_model
        self.d_embed = d_model  # For compatibility with RATE
        self.num_layers = n_layer
        self.num_heads = n_head
        self.d_head = d_model // n_head
        self.env_name = env_name
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.memory_size = memory_size
        self.padding_idx = padding_idx
        self.norm_type = norm_type
        # MoE parameters
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.use_swiglu = use_swiglu

        self.use_shared_expert = use_shared_expert
        self.n_shared_experts = n_shared_experts or 1
        self.shared_d_ff = shared_d_ff
        self.routed_d_ff = routed_d_ff
        
        # Set up dtype
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16
        }
        self.dtype = dtype_map.get(dtype, torch.float32)  # default to float32 if not specified
        
        # Set up sequence format
        valid_formats = ["s", "sa", "sra", "sr"]
        if sequence_format not in valid_formats:
            raise ValueError(f"Invalid sequence_format '{sequence_format}'. Must be one of {valid_formats}")
        self.sequence_format = sequence_format
        
        self.memory_init_std = memory_init_std
        self.detach_memory = detach_memory
        self.use_causal_self_attn_mask = use_causal_self_attn_mask
        self.use_lru = use_lru
        self.lru_blend_alpha = lru_blend_alpha
        self.pos_type = pos_type
        self.train_stride = train_stride
        self.max_seq_len = max_seq_len

        self.mem_tokens = None
        self.attn_map = None
        
        # Initialize encoders and decoders like in RATE
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_model).obs_encoder
        self.action_embeddings = ActEncoder(self.env_name, act_dim, self.d_model).act_encoder
        self.ret_emb = RTGEncoder(self.env_name, self.d_model).rtg_encoder
        self.head = ActDecoder(self.env_name, act_dim, self.d_model).act_decoder
        
        
        if self.pos_type == "relative":
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.pos_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEmbedding(self.d_model)
        elif self.pos_type == "learnable":
            self.pos_emb = LearnablePositionalEmbedding(max_seq_len, self.d_model)
        elif self.pos_type == "rope":
            # RoPE works on head dimension, not full model dimension
            self.pos_emb = RoPEPositionalEmbedding(self.d_head, max_seq_len)
        elif self.pos_type == "yarn":
            # YaRN works on head dimension, following the ICLR 2024 paper
            L_train = max_seq_len  # Original training sequence length
            L_ext = max_seq_len * 4  # 4x context extension
            m = min(128, self.d_head // 4) if self.d_head >= 32 else self.d_head // 2
            
            self.pos_emb = YaRNPositionalEmbedding(
                head_dim=self.d_head,
                L_train=L_train,
                L_ext=L_ext,
                base=10000.0,
                m=m,
                beta=32.0
            )
        elif self.pos_type == "alibi":
            # ALiBi works with attention heads, not model dimension
            self.pos_emb = ALiBiPositionalEmbedding(self.num_heads)
        else:
            raise ValueError(f"Unknown pos_encoding type: {self.pos_type}. Supported types: 'relative', 'sinusoidal', 'learnable', 'rope', 'yarn', 'alibi'")
        
        if self.pos_type == "relative":
            self.r_w_bias = nn.Parameter(torch.zeros(self.num_heads, self.d_head))
            self.r_r_bias = nn.Parameter(torch.zeros(self.num_heads, self.d_head))
        else:
            self.register_parameter("r_w_bias", None)
            self.register_parameter("r_r_bias", None)
        
        # Store memory_dropout for layers
        self.memory_dropout = memory_dropout if memory_dropout is not None else dropout
        
        self.layers = nn.ModuleList([MATLLayer(
            d_model, d_ff, n_head, memory_size, self.pos_type,
            dropout, dropatt, pre_lnorm, max_seq_len,
            memory_init_std, use_lru, lru_blend_alpha, self.memory_dropout,
            self.norm_type,
            # MoE
            use_moe=self.use_moe,
            num_experts=self.num_experts,
            top_k=self.top_k,                     # TOTAL K
            expert_dropout=self.expert_dropout,
            load_balancing_loss_coef=self.load_balancing_loss_coef,
            use_swiglu=self.use_swiglu,
            # DeepSeek-style
            use_shared_expert=self.use_shared_expert,
            n_shared_experts=self.n_shared_experts,
            shared_d_ff=self.shared_d_ff,
            routed_d_ff=self.routed_d_ff
        ) for _ in range(self.num_layers)])
        
        self.drop = nn.Dropout(dropout)
        self._init_parameters()
    
    def _init_parameters(self):
        if self.r_w_bias is not None:
            nn.init.normal_(self.r_w_bias, std=0.02)
            nn.init.normal_(self.r_r_bias, std=0.02)
        
    def init_memory(self, batch_size: int, device: torch.device) -> List[MemoryState]:
        # Initialize memory with the model's dtype
        return [layer.init_memory(batch_size, device) for layer in self.layers]
    
    def encode_actions(self, actions):
        """Encode actions using the action encoder."""
        use_long = False
        for name, module in self.action_embeddings.named_children():
            if isinstance(module, nn.Embedding):
                use_long = True
        if use_long:
            actions = actions.to(dtype=torch.long, device=actions.device)
            actions = torch.where(
                actions == self.padding_idx,
                torch.tensor(self.act_dim, device=actions.device),
                actions,
            )
            action_embeddings = self.action_embeddings(actions).squeeze(2)
        else:
            action_embeddings = self.action_embeddings(actions)

        return action_embeddings

    def reshape_states(self, states):
        """Reshape states for different input formats."""
        reshape_required = False

        if len(states.shape) == 5:
            reshape_required = True
            B, B1, C, H, W = states.shape
        elif len(states.shape) == 6:
            reshape_required = True
            B, B1, _, C, H, W = states.shape
        else:
            B, B1, _ = states.shape
        
        if reshape_required:
            states = states.reshape(-1, C, H, W).to(dtype=self.dtype).contiguous()

        return B, B1, states, reshape_required
    
    def get_sequence_length_multiplier(self):
        """Get the multiplier for sequence length based on sequence format."""
        format_multipliers = {
            "s": 1,    # state only
            "sa": 2,   # state + action
            "sra": 3,  # state + rtg + action
            "sr": 2    # state + rtg
        }
        return format_multipliers[self.sequence_format]
    
    def create_token_sequence(self, state_embeddings, action_embeddings, rtg_embeddings, target, B, B1):
        """Create token sequence based on sequence format."""
        multiplier = self.get_sequence_length_multiplier()
        
        if self.sequence_format == "s":
            # State only: [s1, s2, s3, ...]
            token_embeddings = torch.zeros((B, B1, self.d_embed), 
                                         dtype=self.dtype, device=state_embeddings.device)
            token_embeddings = state_embeddings.to(dtype=self.dtype)
            
        elif self.sequence_format == "sa":
            # State + Action: [s1, a1, s2, a2, ...]
            token_embeddings = torch.zeros((B, B1*2 - int(target is None), self.d_embed), 
                                         dtype=self.dtype, device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings.to(dtype=self.dtype)
            if action_embeddings is not None:
                token_embeddings[:, 1::2, :] = action_embeddings[:,-B1 + int(target is None):,:].to(dtype=self.dtype)
                
        elif self.sequence_format == "sr":
            # State + RTG: [s1, r1, s2, r2, ...]
            token_embeddings = torch.zeros((B, B1*2, self.d_embed), 
                                         dtype=self.dtype, device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings.to(dtype=self.dtype)
            token_embeddings[:, 1::2, :] = rtg_embeddings.to(dtype=self.dtype)
            
        elif self.sequence_format == "sra":
            # State + RTG + Action: [r1, s1, a1, r2, s2, a2, ...]
            if action_embeddings is not None:
                token_embeddings = torch.zeros((B, B1*3 - int(target is None), self.d_embed), 
                                             dtype=self.dtype, device=state_embeddings.device)
                token_embeddings[:, ::3, :] = rtg_embeddings.to(dtype=self.dtype)
                token_embeddings[:, 1::3, :] = state_embeddings.to(dtype=self.dtype)
                token_embeddings[:, 2::3, :] = action_embeddings[:,-B1 + int(target is None):,:].to(dtype=self.dtype)
            else:
                token_embeddings = torch.zeros((B, B1*2, self.d_embed), 
                                             dtype=self.dtype, device=state_embeddings.device)
                token_embeddings[:,::2,:] = rtg_embeddings.to(dtype=self.dtype)
                token_embeddings[:,1::2,:] = state_embeddings.to(dtype=self.dtype)
        
        return token_embeddings
    
    def extract_action_predictions(self, logits, actions):
        """Extract action predictions based on sequence format.
        
        Following GPT logic: predict actions from state token positions.
        """
        if self.sequence_format == "s":
            # For state-only, predict actions from state positions
            return logits
        elif self.sequence_format == "sa":
            # Sequence: [s1, a1, s2, a2, ...] - predict actions from state positions (0, 2, 4, ...)
            if actions is not None:
                return logits[:, ::2, :]   # Extract predictions from state positions
            else:
                # Handle edge case: if we only have states without actions (e.g., first validation step)
                # Return predictions from state positions, but ensure we don't return empty sequence
                if logits.size(1) > 1:
                    return logits[:, 1:, :]    # Extract predictions after states
                else:
                    # Only one token (single state), predict from that token
                    return logits
        elif self.sequence_format == "sr":
            # Sequence: [s1, r1, s2, r2, ...] - predict from state positions (0, 2, 4, ...)
            return logits[:, ::2, :]  # Extract state positions
        elif self.sequence_format == "sra":
            # Sequence: [r1, s1, a1, r2, s2, a2, ...] - predict actions from state positions (1, 4, 7, ...)
            if actions is not None:
                return logits[:, 1::3, :]  # Extract predictions from state positions
            else:
                # Handle edge case: if we only have RTG+state without actions
                if logits.size(1) > 1:
                    return logits[:, 1:, :]    # Extract predictions after RTGs
                else:
                    # Only one token, predict from that token
                    return logits
        
        return logits
    
    def forward(
        self, 
        states, 
        actions, 
        rtgs, 
        target, 
        timesteps, 
        *mems, 
        mem_tokens=None, 
        masks=None, 
        hidden=None,
        memory_states=None,
        pos_offset=0,
        **kwargs
    ):
        """Forward pass through MATL model.
        
        Args:
            states: State observations
            actions: Actions
            rtgs: Return-to-go values
            target: Target actions
            timesteps: Timestep indices
            mems: Memory states (MATL uses different memory format)
            mem_tokens: Memory tokens (not used in MATL, for compatibility)
            masks: Attention masks
            hidden: Hidden state for recurrent models
            
        Returns:
            dict: Contains logits, new memory cache, and updated memory tokens
        """
        
            
        B, B1, states, reshape_required = self.reshape_states(states)
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        rtg_embeddings = self.ret_emb(rtgs)

        # Encode actions if needed
        action_embeddings = None
        if actions is not None and self.sequence_format in ["sa", "sra"]:
            action_embeddings = self.encode_actions(actions)
        
        # Create token sequence based on format
        token_embeddings = self.create_token_sequence(
            state_embeddings, action_embeddings, rtg_embeddings, target, B, B1
        )

        tok = token_embeddings
        bsz, seq_len = tok.size(0), tok.size(1)
        qlen = seq_len

        # # absolute positions for this window, newest first # TODO: it was before
        # tok_pos = torch.arange(
        #     pos_offset + qlen - 1,
        #     pos_offset - 1,
        #     -1,
        #     device=tok.device,
        #     dtype=torch.long
        # )
        # absolute positions for this window
        if self.pos_type in ["rope", "yarn"]:
            # For RoPE/YaRN, use monotonically increasing absolute indices
            tok_pos = torch.arange(pos_offset, pos_offset + qlen, device=tok.device, dtype=torch.long)
        else:
            # For relative attention, use newest-first order (as originally designed)
            tok_pos = torch.arange(
                pos_offset + qlen - 1,
                pos_offset - 1,
                -1,
                device=tok.device,
                dtype=torch.long
            )

        # Create causal mask for current tokens if enabled
        attn_mask = None
        if self.use_causal_self_attn_mask:
            attn_mask = torch.triu(
                torch.ones(qlen, qlen, device=tok.device, dtype=torch.bool), diagonal=1
            )
            if self.pos_type == 'relative':
                attn_mask = attn_mask.unsqueeze(-1)  # [qlen, qlen, 1]
        
        pos_emb = None
        pos_embeddings = None  # For RoPE/YaRN embeddings
        if self.pos_type == "relative":
            tok = tok.transpose(0, 1)  # [T,B,D] - seq_first format
            
            pos_seq = torch.arange(pos_offset + qlen - 1, pos_offset - 1, -1.0, 
                                 device=tok.device, dtype=tok.dtype)
            pos_emb = self.pos_emb(pos_seq).to(dtype=tok.dtype)  # [qlen, 1, D]
            core_out = self.drop(tok)
            pos_emb = self.drop(pos_emb)
        elif self.pos_type == "rope":
            # For RoPE, don't add positional embeddings to tokens
            # Pass RoPE embeddings to attention layers instead
            pos_embeddings = self.pos_emb
            core_out = self.drop(tok).transpose(0, 1)  # [T, B, D]
        elif self.pos_type == "yarn":
            # For YaRN, don't add positional embeddings to tokens
            # Pass YaRN embeddings to attention layers instead
            pos_embeddings = self.pos_emb
            core_out = self.drop(tok).transpose(0, 1)  # [T, B, D]
        elif self.pos_type == "alibi":
            # For ALiBi, don't add positional embeddings to tokens
            # Pass ALiBi embeddings to attention layers instead
            pos_embeddings = self.pos_emb
            core_out = self.drop(tok).transpose(0, 1)  # [T, B, D]
        else:
            if self.pos_type == "learnable":
                pos_indices = torch.arange(seq_len, device=tok.device)
                pos_indices = torch.clamp(pos_indices, 0, self.max_seq_len - 1)
                pos_emb_abs = self.pos_emb(pos_indices).to(dtype=tok.dtype).unsqueeze(0)
            else:
                pos_emb_abs = self.pos_emb(seq_len).to(tok.device, dtype=tok.dtype).unsqueeze(0)
                
            tok = tok + pos_emb_abs
            core_out = self.drop(tok).transpose(0, 1)  # [T, B, D]
        
        updated_memory_states = []
        total_aux_loss = None
        layer_aux_losses = []
        
        for i, layer in enumerate(self.layers):
            # Memory detaching is handled in the trainer, not here
            layer_memory = memory_states[i]
            
            core_out, updated_mem, layer_aux_loss = layer(
                core_out,             # [T, B, D]
                layer_memory,         # MemoryState
                tok_pos,              # [T] integer positions
                pos_emb,              # [T,1,D] for self-attn
                self.r_w_bias,
                self.r_r_bias,
                attn_mask,
                pos_embeddings        # RoPE/YaRN embeddings for attention
            )
            updated_memory_states.append(updated_mem)
            
            # Collect auxiliary losses from MoE
            if layer_aux_loss is not None:
                layer_aux_losses.append(layer_aux_loss)
        
        # Sum all auxiliary losses
        if layer_aux_losses:
            total_aux_loss = sum(layer_aux_losses)
        
        core_out = core_out.transpose(0, 1)  # [B, T, D]
        
        # Use head decoder for action prediction like in RATE
        logits = self.head(core_out)
        
        # Extract action predictions based on sequence format
        logits = self.extract_action_predictions(logits, actions)
        
        output = {
            'logits': logits,
            'memory_states': updated_memory_states,
            'new_mems': None,  # For compatibility with RATE
            'mem_tokens': None,  # MATL doesn't use mem_tokens
            'aux_loss': total_aux_loss  # MoE auxiliary loss
        }
        
        return output
    
    def get_moe_stats(self) -> dict:
        """Get MoE usage statistics and parameter counts"""
        if not self.use_moe:
            return {"moe_enabled": False}
        
        total_params = sum(p.numel() for p in self.parameters())
        moe_params = 0
        expert_params = 0
        
        for layer in self.layers:
            # Count MoE parameters
            moe_params += sum(p.numel() for p in layer.token_ffn.parameters())
            moe_params += sum(p.numel() for p in layer.memory_ffn.parameters())
            
            # Count expert parameters specifically
            if hasattr(layer.token_ffn.core_net, 'experts'):
                expert_params += sum(p.numel() for e in layer.token_ffn.core_net.experts for p in e.parameters())
                if getattr(layer.token_ffn.core_net, 'shared_expert', None) is not None:
                    expert_params += sum(p.numel() for p in layer.token_ffn.core_net.shared_expert.parameters())

            if hasattr(layer.memory_ffn.core_net, 'experts'):
                expert_params += sum(p.numel() for e in layer.memory_ffn.core_net.experts for p in e.parameters())
                if getattr(layer.memory_ffn.core_net, 'shared_expert', None) is not None:
                    expert_params += sum(p.numel() for p in layer.memory_ffn.core_net.shared_expert.parameters())
        
        return {
            "moe_enabled": True,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "total_parameters": total_params,
            "moe_parameters": moe_params,
            "expert_parameters": expert_params,
            "expert_param_ratio": expert_params / total_params if total_params > 0 else 0,
            "use_swiglu": self.use_swiglu,
            "load_balancing_coef": self.load_balancing_loss_coef
        } 