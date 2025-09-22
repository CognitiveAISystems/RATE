# Adopted from: https://github.com/WangJinCheng1998/LONG-SHORT-DECISION-TRANSFORMER

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

from RATE.env_encoders import ObsEncoder, ActEncoder, RTGEncoder, ActDecoder

class DynamicConvolution(nn.Module):
    def __init__(
        self,
        wshare,
        n_feat,
        dropout_rate,
        kernel_size,
        use_kernel_mask=False,
        use_bias=False,
    ):
        super(DynamicConvolution, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.attn = None

        self.linear1 = nn.Linear(n_feat, n_feat * 2)
        self.linear2 = nn.Linear(n_feat, n_feat)
        self.linear_weight = nn.Linear(n_feat, self.wshare * 1 * kernel_size)
        init.xavier_uniform_(self.linear_weight.weight)
        self.act = nn.GLU()

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat))
            self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, query, key, value, count, mask):
        x = query
        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size
        
        x = self.linear1(x)
        x = self.act(x)
        
        weight = self.linear_weight(x)
        weight = F.dropout(weight, self.dropout_rate, training=self.training)
        weight = weight.view(B, T, H, k).transpose(1, 2).contiguous()
        
        weight_new = torch.zeros(B * H * T * (T + k - 1), dtype=weight.dtype)
        weight_new = weight_new.view(B, H, T, T + k - 1).fill_(float("-inf"))
        weight_new = weight_new.to(x.device)
        
        weight_new.as_strided(
            (B, H, T, k), ((T + k - 1) * T * H, (T + k - 1) * T, T + k, 1)
        ).copy_(weight)
        
        weight_new = weight_new.narrow(-1, int((k - 1) / 2), T)
        
        if self.use_kernel_mask:
            kernel_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)
            weight_new = weight_new.masked_fill(kernel_mask == 0.0, float("-inf"))
            
        weight_new = F.softmax(weight_new, dim=-1)
        self.attn = weight_new
        weight_new = weight_new.view(B * H, T, T)

        x = x.transpose(1, 2).contiguous()
        x = x.view(B * H, int(C / H), T).transpose(1, 2)
        x = torch.bmm(weight_new, x)
        x = x.transpose(1, 2).contiguous().view(B, C, T)

        if self.use_bias:
            x = x + self.bias.view(1, -1, 1)
        x = x.transpose(1, 2)

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1, -2)
            x = x.masked_fill(mask == 0, 0.0)

        x = self.linear2(x)
        return x

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, kernelsize, dropk, convdim):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T
        self.drop_p = drop_p
        Dimratio = int(convdim)
        self.attdim = h_dim - Dimratio
        self.convdim = Dimratio

        if self.attdim != 0:
            self.q_net = nn.Linear(self.attdim, self.attdim)
            self.k_net = nn.Linear(self.attdim, self.attdim)
            self.v_net = nn.Linear(self.attdim, self.attdim)

            self.fmlp = nn.Sequential(
                nn.Linear(h_dim, h_dim//2),
                nn.ReLU(),
            )
            self.fl1 = nn.Linear(h_dim//2, self.attdim)
            self.fl2 = nn.Linear(h_dim//2, self.convdim)

        self.proj_net = nn.Linear(h_dim, h_dim)
        self.dropk = dropk
        self.att_drop = nn.Dropout(dropk)
        self.proj_drop = nn.Dropout(drop_p)
        
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones)
        self.kernelsize = kernelsize
    
        mask = mask.view(1, 1, max_T, max_T)
        self.register_buffer('mask', mask)
        
        self.Dynamicconv_layer = DynamicConvolution(
            wshare=self.convdim//4,
            n_feat=self.convdim,
            dropout_rate=0.2,
            kernel_size=self.kernelsize,
            use_kernel_mask=True,
            use_bias=True
        )

    def forward(self, x):
        B, T, D = x.shape

        x1, x2 = torch.split(x, [self.attdim, self.convdim], dim=-1)
        
        if self.attdim != 0:
            attention_future = torch.jit.fork(self.attention_branch, x1, B, T)
            conv_future = torch.jit.fork(self.conv_branch, x2)

            attention_output = torch.jit.wait(attention_future)
            conv_output = torch.jit.wait(conv_future)

            attention2 = torch.cat((attention_output, conv_output), dim=-1)
            output = attention2
            Xbar = self.fmlp(attention2)

            X1 = self.fl1(Xbar)
            X2 = self.fl2(Xbar)
            Z = torch.cat((X1, X2), dim=-1)
            Xf = F.softmax(Z, dim=-1)
            output = Xf * attention2
        else:
            conv_future = torch.jit.fork(self.conv_branch, x2)
            output = torch.jit.wait(conv_future)

        out = self.proj_drop(self.proj_net(output))
        return out
    
    def attention_branch(self, x1, B, T):
        N, D = self.n_heads, x1.size(2) // self.n_heads

        q = self.q_net(x1).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x1).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x1).view(B, T, N, D).transpose(1, 2)

        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        
        attention = self.att_drop(weights @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)
        return attention
    
    def conv_branch(self, x2):
        conv_output = self.Dynamicconv_layer(x2, x2, x2, 0, mask=None)
        return conv_output

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, kernelsize, dropk, convdim):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p, kernelsize, dropk, convdim)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x

class LongShortDecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_layer=3,
        n_head=2,
        d_model=128,
        d_head=None,
        d_inner=None,
        dropout=0.1,
        dropatt=0.1,
        kernel_size=11,
        convdim=64,
        env_name='mujoco',
        max_ep_len=1000,
        padding_idx=None,
        **kwargs
    ):
        super().__init__()

        self.d_embed = d_model
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head if d_head is not None else d_model // n_head
        self.env_name = env_name
        self.act_dim = act_dim
        self.padding_idx = padding_idx
        self.mem_tokens = None
        self.attn_map = None

        # Initialize encoders
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        self.action_embeddings = ActEncoder(self.env_name, act_dim, self.d_embed).act_encoder
        self.ret_emb = RTGEncoder(self.env_name, self.d_embed).rtg_encoder
        self.embed_timestep = nn.Embedding(max_ep_len, self.d_embed)
        self.embed_ln = nn.LayerNorm(self.d_embed)
        self.drop = nn.Dropout(dropout)

        # Initialize transformer blocks
        blocks = []
        dk = [0.1] * n_layer  # dropout rates for attention branch
        for i in range(n_layer):
            blocks.append(Block(
                h_dim=d_model,
                max_T=max_ep_len * 3,  # For RTG, state, action sequence
                n_heads=n_head,
                drop_p=dropout,
                kernelsize=kernel_size,
                dropk=dk[i],
                convdim=convdim
            ))
        self.transformer = nn.Sequential(*blocks)

        # Initialize decoder
        self.head = ActDecoder(self.env_name, act_dim, self.d_embed).act_decoder

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            init.xavier_uniform_(m.weight)

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

    def reshape_states(self, states):
        reshape_required = False
        use_long = False
        for name, module in self.action_embeddings.named_children():
            if isinstance(module, nn.Embedding):
                use_long = True

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

        if use_long:
            states = states.squeeze(2)

        return B, B1, states, reshape_required

    def forward(self, states, actions=None, rtgs=None, target=None, timesteps=None, mem_tokens=None, masks=None, hidden=None, *args, **kwargs):
        B, B1, states, reshape_required = self.reshape_states(states)
        
        # Get state embeddings
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        
        # Get RTG embeddings
        rtg_embeddings = self.ret_emb(rtgs)
        print("timesteps shape: ", timesteps.shape)
        if len(timesteps.shape) == 4: timesteps = timesteps.squeeze(-2)
        time_embeddings = self.embed_timestep(timesteps)
        if len(time_embeddings.shape) == 4: time_embeddings = time_embeddings.squeeze(-2)
        print(f"Time embeddings shape: {time_embeddings.shape}, RTG embeddings shape: {rtg_embeddings.shape}") # torch.Size([64, 15, 128])
        
        # Prepare token sequence
        if actions is not None:
            action_embeddings = self.encode_actions(actions)
            
            # Format tokens as in RATE: alternate rtg, state, action
            token_embeddings = torch.zeros((B, B1*3 - int(target is None), self.d_embed), dtype=torch.float32, device=state_embeddings.device)
            print(f"\n\n\nToken embeddings shape: {token_embeddings.shape}, rtg_embeddings shape: {rtg_embeddings.shape}, state_embeddings shape: {state_embeddings.shape}, action_embeddings shape: {action_embeddings.shape}, time_embeddings shape: {time_embeddings.shape}")
            token_embeddings[:, ::3, :] = rtg_embeddings + time_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings + time_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:,-B1 + int(target is None):,:] + time_embeddings[:,-B1 + int(target is None):,:]
        else:
            # If no actions, alternate only rtg and state
            token_embeddings = torch.zeros((B, B1*2, self.d_embed), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::2, :] = rtg_embeddings + time_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings + time_embeddings

        # # Add time embeddings
        # if timesteps is not None:
        #     time_embeddings = self.embed_timestep(timesteps.squeeze(-1))
        #     print(f"Time embeddings shape: {time_embeddings.shape}") # torch.Size([64, 15, 128])
        #     print(f"Token embeddings shape: {token_embeddings.shape}") # torch.Size([64, 45, 128])
        #     token_embeddings = token_embeddings + time_embeddings

#         # Add time embeddings
#         if timesteps is not None:
#             time_embeddings = self.embed_timestep(timesteps.squeeze(-1))
            
#             # Modify time embeddings to repeat each timestep three times (format 111,222,333)
#             # First, determine the pattern of repetition based on token structure
#             if actions is not None:
#                 # For rtg, state, action pattern (every 3 tokens)
#                 expanded_time_embeddings = torch.repeat_interleave(time_embeddings, 3, dim=1)
#                 # Truncate to match token_embeddings shape if needed
#                 expanded_time_embeddings = expanded_time_embeddings[:, :token_embeddings.shape[1], :]
#             else:
#                 # For rtg, state pattern (every 2 tokens)
#                 expanded_time_embeddings = torch.repeat_interleave(time_embeddings, 2, dim=1)
#                 # Truncate to match token_embeddings shape if needed
#                 expanded_time_embeddings = expanded_time_embeddings[:, :token_embeddings.shape[1], :]
                
#             token_embeddings = token_embeddings + expanded_time_embeddings

        # Apply dropout and layer norm
        token_embeddings = self.drop(token_embeddings)
        token_embeddings = self.embed_ln(token_embeddings)

        # Process through transformer
        hidden = self.transformer(token_embeddings)

        # Get action predictions
        logits = self.head(hidden)

        if actions is not None:
            logits = logits[:, 1::3, :]
        else:
            logits = logits[:, 1:, :]

        output = {
            'logits': logits,
            'new_mems': None,  # LSDT doesn't use memory
            'mem_tokens': None,  # LSDT doesn't use memory tokens
            'hidden': None  # LSDT doesn't use hidden state
        }

        return output

    def reset_hidden(self, batch_size=None, device=None):
        """For compatibility with other models, but LSDT doesn't use hidden state"""
        return None
