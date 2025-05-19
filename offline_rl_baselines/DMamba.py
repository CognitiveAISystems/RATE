# Adopted from: https://github.com/Toshihiro-Ota/decision-mamba
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba

from RATE.env_encoders import ObsEncoder, ActEncoder, RTGEncoder, ActDecoder


class DMamba(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_layer=6,
        n_head=8,
        d_model=128,
        dropout=0.1,
        dropatt=0.0,
        tgt_len=None,
        max_ep_len=1000,
        env_name='tmaze',
        padding_idx=None,
        token_mixer='mamba',  # mamba, mamba-min, attn, conv, conv-attn
        window_size=4,  # For conv mixer
        conv_proj=True,  # For conv mixer
        **kwargs
    ):
        super(DMamba, self).__init__()
        
        self.d_embed = d_model
        self.d_model = d_model
        self.token_mixer = token_mixer
        self.padding_idx = padding_idx
        self.act_dim = act_dim
        self.model_type = 'reward_conditioned'
        self.block_size = tgt_len if tgt_len is not None else 1024
        self.n_layer = n_layer
        self.max_timestep = max_ep_len
        self.n_embd = d_model
        self.n_head = n_head
        self.window_size = window_size
        self.conv_proj = conv_proj
        self.embd_pdrop = dropout
        self.resid_pdrop = dropout
        self.attn_pdrop = dropatt
        self.env_name = env_name
        self.mem_tokens = None
        
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        self.action_embeddings = ActEncoder(self.env_name, act_dim, self.d_embed).act_encoder
        self.ret_emb = RTGEncoder(self.env_name, self.d_embed).rtg_encoder
        self.head = ActDecoder(self.env_name, act_dim, self.d_embed).act_decoder
        
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size + 1, d_model))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, max_ep_len + 1, d_model))
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[Block(self, index) for index in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_block_size(self):
        return self.block_size
    
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith('D') or pn.endswith('A_log') or fpn.endswith('norm_mamba.weight'):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay and no_decay"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not split"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )
        
        return optimizer
    
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
                    torch.tensor(self.act_dim, device=actions.device),
                    actions,
                )
            action_embeddings = self.action_embeddings(actions).squeeze(2)
        else:
            action_embeddings = self.action_embeddings(actions)

        return action_embeddings

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
    
    def forward(self, states, actions, rtgs, target, timesteps, *mems, mem_tokens=None, masks=None, hidden=None):
        """
        Forward pass, compatible with RATE's trainer.py expectations
        
        Args:
            states: batch of states
            actions: batch of actions (or None for initial prediction)
            rtgs: batch of return-to-go
            target: target values for loss computation (usually matches actions)
            timesteps: current timesteps in episode
            
        Returns:
            Dict with 'logits' and placeholders for compatibility with RATE
        """
        # Process input dimensions
        B, B1, states, reshape_required = self.reshape_states(states)
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        
        rtg_embeddings = self.ret_emb(rtgs)
        
        if actions is not None:
            action_embeddings = self.encode_actions(actions)
            
            # Create token embeddings in the correct order (rtg, state, action)
            token_embeddings = torch.zeros(
                (B, B1*3 - int(target is None), self.d_embed),
                dtype=torch.float32, 
                device=state_embeddings.device
            )
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -B1 + int(target is None):, :]
        else:
            # Initial prediction (actions not yet present)
            token_embeddings = torch.zeros(
                (B, B1*2, self.d_embed),
                dtype=torch.float32, 
                device=state_embeddings.device
            )
            token_embeddings[:, ::2, :] = rtg_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings
        
        # Add positional embeddings
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        # print(timesteps.shape, all_global_pos_emb.shape, self.pos_emb.shape) # 64x9, 64x1001x64
        
        # 64x9x64 + 64x27x64
        # position_embeddings = torch.gather(
        #     all_global_pos_emb, 
        #     1, 
        #     torch.repeat_interleave(timesteps.unsqueeze(-1), self.d_model, dim=-1)
        # ) + self.pos_emb[:, :token_embeddings.shape[1], :]

        # time_pos_emb = torch.gather(
        #     all_global_pos_emb, 
        #     1, 
        #     torch.repeat_interleave(timesteps.unsqueeze(-1), self.d_model, dim=-1)
        # )  # [64, 9, 64]

        # # Expand each timestep, repeating it 3 times
        # expanded_time_pos = torch.repeat_interleave(time_pos_emb, 3, dim=1)  # [64, 27, 64]

        # # Trim to the desired size to match token_embeddings
        # expanded_time_pos = expanded_time_pos[:, :token_embeddings.shape[1], :]

        # # Add positional embeddings
        # position_embeddings = expanded_time_pos + self.pos_emb[:, :token_embeddings.shape[1], :]

        
        # Apply transformer blocks
        # x = self.drop(token_embeddings + position_embeddings)
        x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Extract action predictions from output
        if actions is not None:
            logits = logits[:, 1::3, :]  # Keep only predictions from state embeddings
        else:
            logits = logits[:, 1:, :]
        
        # Return in the format expected by trainer.py in RATE
        output = {
            'logits': logits,
            'new_mems': None,  # Placeholder for compatibility
            'mem_tokens': None,  # Placeholder for compatibility
        }
        
        return output


# Token mixing blocks
class CausalSelfAttention(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # projection of key, query, value
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
            .view(1, 1, config.block_size + 1, config.block_size + 1)
        )
        self.n_head = config.n_head
    
    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        
        # Compute query, key, value for all heads
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Self-attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Compute output
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        
        return y


class Convolution(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.window_size = config.window_size
        hidden_size = config.n_embd
        self.conv_proj = config.conv_proj
        
        # Separate convolutions for rtg, observation, and action
        self.rtg_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.obs_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.act_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        
        if config.conv_proj:
            self.fc = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Padding input tensor
        padded_tensor = torch.nn.functional.pad(x, (0, 0, self.window_size - 1, 0)).transpose(1, 2)
        
        # Apply convolutions separately to positions rtg, obs and action
        rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
        obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
        act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]
        
        # Combine outputs
        conv_tensor = torch.zeros((x.shape[0], x.shape[2], x.shape[1]), device=x.device)
        conv_tensor[:, :, ::3] = rtg_conv_tensor
        conv_tensor[:, :, 1::3] = obs_conv_tensor
        conv_tensor[:, :, 2::3] = act_conv_tensor
        conv_tensor = conv_tensor.transpose(1, 2)
        
        # Apply projection if enabled
        if self.conv_proj:
            conv_tensor = self.dropout(self.fc(conv_tensor))
        
        return conv_tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


@dataclass
class ModelArgs:
    d_model: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        
        # Initialize dt projections
        dt_init_std = args.dt_rank**-0.5 * args.dt_scale
        if args.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif args.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # Initialize dt offset
        dt = torch.exp(torch.rand(args.d_inner) * 
                      (math.log(args.dt_max) - math.log(args.dt_min)) + 
                      math.log(args.dt_min)).clamp(min=args.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # Parameters A and D
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
    
    def forward(self, x):
        b, l, d = x.shape
        
        # Input projection and gating
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        
        # 1D convolution
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        
        # State space model
        y = self.ssm(x)
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        return output
    
    def ssm(self, x):
        d_in, n = self.A_log.shape
        
        # Compute state space parameters
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        # Input projection for delta, B, C
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        # Run selective scanning
        y = self.selective_scan(x, delta, A, B, C, D)
        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        n = A.shape[1]
        
        # Discretization
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Sequential scanning (for simplicity, but slower than parallel implementation)
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        
        # Add residual connection
        y = y + u * D
        return y


class Block(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.token_mixer = config.token_mixer
        self.n_layer = config.n_layer
        self.index = index
        
        if self.token_mixer == 'attn':
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config, index)
        elif self.token_mixer == 'conv':
            self.lnc = nn.LayerNorm(config.n_embd)
            self.conv = Convolution(config, index)
        elif self.token_mixer == 'conv-attn':
            if self.index < self.n_layer - 1:
                self.lnc = nn.LayerNorm(config.n_embd)
                self.conv = Convolution(config, index)
            else:
                self.ln1 = nn.LayerNorm(config.n_embd)
                self.attn = CausalSelfAttention(config, index)
        elif self.token_mixer == 'mamba':
            self.norm_mamba = nn.LayerNorm(config.n_embd)
            self.mamba = Mamba(config.n_embd)
        elif self.token_mixer == 'mamba-min':
            self.norm_mamba = RMSNorm(config.n_embd)
            self.mamba = MambaBlock(ModelArgs(d_model=config.n_embd))
        else:
            raise NotImplementedError
        
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp_channels = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    
    def forward(self, x):
        if self.token_mixer == 'attn':
            x = x + self.attn(self.ln1(x))
        elif self.token_mixer == 'conv':
            x = x + self.conv(self.lnc(x))
        elif self.token_mixer == 'conv-attn':
            if self.index < self.n_layer - 1:
                x = x + self.conv(self.lnc(x))
            else:
                x = x + self.attn(self.ln1(x))
        elif self.token_mixer == 'mamba' or self.token_mixer == 'mamba-min':
            x = x + self.mamba(self.norm_mamba(x))
        else:
            raise NotImplementedError
        
        x = x + self.mlp_channels(self.ln2(x))
        return x