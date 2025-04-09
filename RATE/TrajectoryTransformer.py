# RATE/TrajectoryTransformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from RATE.env_encoders import ObsEncoder, ActEncoder, RTGEncoder, ActDecoder

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, dropatt, block_size=1024):
        super().__init__()
        assert d_model % n_head == 0
        
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        # key, query, value projections
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # regularization
        self.attn_drop = nn.Dropout(dropatt)
        self.resid_drop = nn.Dropout(dropout)
        
        # output projection
        self.proj = nn.Linear(d_model, d_model)
        
        # causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        
        # calculate query, key, values for all heads
        k = self.key(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, dropatt, d_inner=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, dropatt)
        
        if d_inner is None:
            d_inner = 4 * d_model
            
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TrajectoryTransformer(nn.Module):
    def __init__(
            self,
            state_dim,
            act_dim,
            n_layer=8,
            n_head=8,
            n_head_ca=0,  # Not used, but needed for interface compatibility
            d_model=128,
            d_head=None,  # Not used, but needed for interface compatibility
            d_inner=None,
            dropout=0.1,
            dropatt=0.1,
            pre_lnorm=False,  # Not used, but needed for interface compatibility
            tgt_len=None,  # Not used, but needed for interface compatibility
            ext_len=None,  # Not used, but needed for interface compatibility
            mem_len=None,  # Not used, but needed for interface compatibility
            num_mem_tokens=None,  # Not used, but needed for interface compatibility
            read_mem_from_cache=False,  # Not used, but needed for interface compatibility
            mem_at_end=True,  # Not used, but needed for interface compatibility
            same_length=False,  # Not used, but needed for interface compatibility
            clamp_len=-1,  # Not used, but needed for interface compatibility
            sample_softmax=-1,  # Not used, but needed for interface compatibility
            max_ep_len=1000,
            env_name='tmaze',
            use_gate=False,  # Not used, but needed for interface compatibility
            use_stable_version=False,  # Not used, but needed for interface compatibility
            mrv_act='no_act',  # Not used, but needed for interface compatibility
            skip_dec_ffn=False,  # Not used, but needed for interface compatibility
            padding_idx=None,
            **kwargs
        ):
        super(TrajectoryTransformer, self).__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer  # Явно сохраняем n_layer как атрибут класса
        self.d_inner = 4 * d_model if d_inner is None else d_inner
        self.dropout = dropout
        self.max_ep_len = max_ep_len
        self.env_name = env_name
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.padding_idx = padding_idx
        
        # Store for interface compatibility
        self.mem_len = 0  # Trajectory Transformer doesn't use memory length, setting to 0
        self.ext_len = 0  # Trajectory Transformer doesn't use extended length, setting to 0
        self.num_mem_tokens = 0  # Trajectory Transformer doesn't use memory tokens, setting to 0
        self.mem_tokens = None  # Explicitly set mem_tokens to None for compatibility
        
        # Embeddings
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_model).obs_encoder
        self.action_embeddings = ActEncoder(self.env_name, act_dim, self.d_model).act_encoder
        self.ret_emb = RTGEncoder(self.env_name, self.d_model).rtg_encoder
        self.embed_timestep = nn.Embedding(max_ep_len, self.d_model)
        self.embed_ln = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(dropout)
        
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, self.d_model))
        
        # Transformer layers
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_head, dropout, dropatt, self.d_inner) 
            for _ in range(n_layer)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(d_model)
        self.action_decoder = ActDecoder(self.env_name, act_dim, d_model).act_decoder
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode_actions(self, actions):
        use_long = False
        for name, module in self.action_embeddings.named_children():
            if isinstance(module, nn.Embedding):
                use_long = True
        if use_long:
            actions = actions.to(dtype=torch.long, device=actions.device)
            if self.padding_idx is not None:
                actions = torch.where(
                    actions == self.padding_idx,
                    torch.tensor(self.act_dim),
                    actions,
                )
            
            action_embeddings = self.action_embeddings(actions).squeeze(2)
        else:
            action_embeddings = self.action_embeddings(actions)

        return action_embeddings
            
    def forward(self, states, actions=None, returns_to_go=None, timesteps=None,
               attention_mask=None, state_padding_mask=None, action_padding_mask=None,
               return_hidden=False, return_attention_weights=False, mems=None, **kwargs):
        """
        Forward pass of the Trajectory Transformer model.
        
        Args:
            states: (B, T, state_dim) tensor of state representations
            actions: (B, T, act_dim) tensor of action representations (optional for inference)
            returns_to_go: (B, T, 1) tensor of returns-to-go (optional for inference)
            timesteps: (B, T) tensor of timesteps (optional, defaults to range(T))
            attention_mask: Attention mask for the transformer (optional)
            state_padding_mask: Padding mask for states (optional)
            action_padding_mask: Padding mask for actions (optional)
            return_hidden: Whether to return hidden states (default: False)
            return_attention_weights: Whether to return attention weights (default: False)
            mems: Memory for TransformerXL compatibility (not used)
            
        Returns:
            action_preds: (B, T, act_dim) tensor of predicted actions
            outputs: Additional outputs (hidden_states, attentions) if requested
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if timesteps is None:
            timesteps = torch.arange(seq_length, device=states.device).repeat(batch_size, 1)
        
        # Get embeddings for states, actions, returns, and timesteps
        state_embeddings = self.state_encoder(states)
        
        if actions is not None:
            # print('actions', actions.shape, actions)
            # action_embeddings = self.action_embeddings(actions)
            action_embeddings = self.encode_actions(actions)
        else:
            # For inference, we'll use zeros and overwrite later
            action_embeddings = torch.zeros(
                (batch_size, seq_length, self.d_model), 
                device=states.device
            )
            
        if returns_to_go is not None:
            returns_embeddings = self.ret_emb(returns_to_go)
        else:
            # For inference without returns
            returns_embeddings = torch.zeros(
                (batch_size, seq_length, self.d_model),
                device=states.device
            )
        
        # time_embeddings = self.embed_timestep(timesteps)
        
        # Form sequences: [r_0, s_0, a_0, r_1, s_1, a_1, ...]
        sequence = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings],
            dim=2
        ).reshape(batch_size, 3 * seq_length, self.d_model)
        
        # Add position embeddings and apply layer norm
        if seq_length <= 1024:
            # Using pre-computed position embeddings
            position_embeddings = self.pos_emb[:, :3 * seq_length, :]
            sequence = sequence + position_embeddings
        
        sequence = self.embed_ln(sequence)
        sequence = self.drop(sequence)
        
        # Apply transformer blocks
        hidden_states = self.blocks(sequence)
        hidden_states = self.ln_f(hidden_states)
        
        action_preds = hidden_states[:, 2::3, :]  # indices 2, 5, 8, ...
        action_preds = self.action_decoder(action_preds)
        
        attn_maps = []
        for block in self.blocks:
            if hasattr(block.attn, '_attn_map'):
                attn_maps.append(block.attn._attn_map)
        
        if len(attn_maps) > 0:
            self.attn_map = attn_maps[0]
        
        outputs = {'logits': action_preds}
        
        if return_hidden:
            outputs['hidden_states'] = hidden_states
            
        if return_attention_weights:
            outputs['attentions'] = attn_maps
            
        # For interface compatibility with RATE
        outputs['mems'] = None
            
        return outputs
        
    def get_action(self, states, actions, rtg, timesteps, **kwargs):
        """
        Get predicted action for the current step during inference.
        
        Args:
            states: (B, T, state_dim) tensor of state representations
            actions: (B, T, act_dim) tensor of action representations (includes zeros for prediction position)
            rtg: (B, T, 1) tensor of returns-to-go
            timesteps: (B, T) tensor of timesteps
            
        Returns:
            action_preds: (B, 1, act_dim) tensor of predicted actions for the current step
        """
        outputs = self.forward(
            states=states, 
            actions=actions, 
            returns_to_go=rtg, 
            timesteps=timesteps,
            **kwargs
        )
        
        return outputs['logits'][:, -1:]  # return only the last predicted action

    def init_mems(self, device):
        mems = []
        for i in range(self.n_layer + 1):
            empty = torch.empty(0, dtype=torch.float, device=device)
            mems.append(empty)
        
        return mems

    def reshape_states(self, states):
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
            states = states.reshape(-1, C, H, W).type(torch.float32).contiguous()

        return B, B1, states, reshape_required