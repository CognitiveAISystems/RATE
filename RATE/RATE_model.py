import sys
import math
import functools

import numpy as np
 
import torch
import torch.nn as nn
import torch.nn.functional as F

from RATE.utils import LogUniformSampler, sample_logits, ProjectedAdaptiveLogSoftmax
from RATE.blocks import RelPartialLearnableDecoderLayer, PositionalEmbedding
from RATE.env_encoders import ObsEncoder, ActEncoder, RTGEncoder, ActDecoder

class MemTransformerLM(nn.Module):
    def __init__(
        self, 
        state_dim, 
        act_dim,
        n_layer, 
        n_head,
        n_head_ca, 
        d_model, 
        d_head, 
        d_inner,
        dropout, 
        dropatt, 
        pre_lnorm=False,
        tgt_len=None, 
        ext_len=None, 
        mem_len=None, 
        num_mem_tokens=None, 
        read_mem_from_cache=False, 
        mem_at_end=True,
        same_length=False,
        clamp_len=-1, 
        sample_softmax=-1,
        max_ep_len=1000,
        env_name='mujoco',
        use_gate=False,
        use_stable_version=False,
        mrv_act='relu',
        skip_dec_ffn=False,
        padding_idx=None,
        **kwargs
    ):
        
        super(MemTransformerLM, self).__init__()

        self.d_embed = d_model
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.skip_dec_ffn = skip_dec_ffn
        self.env_name = env_name
        self.is_first_segment = True
        self.log_prob = 0
        self.buf = []
        self.embed_timestep = nn.Embedding(max_ep_len, self.d_embed)
        self.embed_ln = nn.LayerNorm(self.d_embed)
        self.drop = nn.Dropout(dropout)
        self.n_layer = n_layer
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = None # tgt_len + ext_len + mem_len + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.read_mem_from_cache = read_mem_from_cache 
        self.mem_at_end = mem_at_end
        self.n_head_ca = n_head_ca
        self.sample_softmax = sample_softmax
        self.same_length = same_length 
        self.clamp_len = clamp_len
        self.padding_idx = padding_idx
        self.act_dim = act_dim

        self._set_mrv_act(mrv_act)
            
        # tmaze, aar, memory_maze, minigrid_memory, vizdoom, atari, mujoco, maniskill-pushcube, popgym(48envs)
        self.state_encoder      = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        self.action_embeddings  = ActEncoder(self.env_name, act_dim, self.d_embed).act_encoder
        self.ret_emb            = RTGEncoder(self.env_name, self.d_embed).rtg_encoder
        self.head               = ActDecoder(self.env_name, act_dim, self.d_embed).act_decoder

        self.init_mem_tokens()

        if self.n_head_ca > 0:
            self.mha_mem_to_mem = MultiHeadAttention(
                d_q=self.d_model,
                d_k=self.d_model, 
                d_v=self.d_model,
                d_model=self.d_model,
                num_heads=self.n_head_ca,
                dropout_p=dropatt,
                is_causal=False,
            )

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    use_gate=use_gate, use_stable_version=use_stable_version,
                    qkw_norm=pre_lnorm, skip_dec_ffn=skip_dec_ffn,# !
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

        self._create_params()

    def _set_mrv_act(self, mrv_act):
        if mrv_act == 'relu':
            self.mrv_act = F.relu
        elif mrv_act == 'leaky_relu':
            self.mrv_act = F.leaky_relu
        elif mrv_act == 'elu':
            self.mrv_act == F.elu
        elif mrv_act == 'tanh':
            self.mrv_act = F.tanh
        elif mrv_act == 'no_act':
            self.mrv_act = None
        else:
            raise NotImplementedError('This MRV activation is not studied')
        
    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
 
    def init_mems(self, device):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=torch.float, device=device)
                mems.append(empty)

            return mems
        else:
            return None

    def init_mem_tokens(self):
        if self.num_mem_tokens == 0:
            self.mem_tokens = None
        else:
            mem_tokens = [torch.randn(1, self.d_model)] * self.num_mem_tokens
            mem_tokens = torch.cat(mem_tokens, dim=0).view(self.num_mem_tokens, 1, -1)
            mem_tokens = torch.nn.Parameter(mem_tokens, requires_grad=True)
            self.register_parameter(param=mem_tokens, name='mem_tokens')

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None: 
            return None

        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
                
        return new_mems

    def _forward(self, word_emb, mems=None, mem_tokens=None):
        bsz, qlen, _ = word_emb.size()
        word_emb = word_emb.permute(1,0,2)
        mlen = mems[0].size(0) if mems is not None else 0

        # Concat with mem_tokens
        if mem_tokens is not None:
            word_emb = torch.cat((mem_tokens, word_emb), dim=0)
            if self.mem_at_end:
                word_emb = torch.cat((word_emb, mem_tokens), dim=0) # (nmt+N*K+nmt, bs, emb_dim)

        qlen = word_emb.shape[0]
        klen = mlen + qlen
        
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()

            if self.num_mem_tokens != 0:
                dec_attn_mask[:self.num_mem_tokens, mlen:mlen+self.num_mem_tokens] = 0
                dec_attn_mask[:self.num_mem_tokens, :mlen] = 1 - int(self.read_mem_from_cache)
                if self.mem_at_end:
                    dec_attn_mask[-self.num_mem_tokens:, -self.num_mem_tokens:] = 0
                    dec_attn_mask[-self.num_mem_tokens:, :mlen] = 1 - int(self.read_mem_from_cache)
            dec_attn_mask = dec_attn_mask[:,:,None]

        hids = []
        pos_seq = torch.arange(
            klen-1, -1, -1.0,
            device=word_emb.device,
            dtype=word_emb.dtype
        )

        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out, self.attn_map = layer(
                core_out, pos_emb, self.r_w_bias,
                self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i,
            )
            hids.append(core_out)

        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems
    
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
    
    def forward(self, states, actions, rtgs, target, timesteps, *mems, mem_tokens=None, masks=None):
        
        if not mems: mems = self.init_mems(states.device)
        B, B1, states, reshape_required = self.reshape_states(states)
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        rtg_embeddings = self.ret_emb(rtgs)

        if actions is not None:
            action_embeddings = self.encode_actions(actions)
            token_embeddings = torch.zeros((B, B1*3 - int(target is None), self.d_embed), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:,-B1 + int(target is None):,:]
        else:
            token_embeddings = torch.zeros((B, B1*2, self.d_embed), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings
            token_embeddings[:,1::2,:] = state_embeddings

        hidden, new_mems = self._forward(token_embeddings, mems=mems, mem_tokens=mem_tokens) #hidden.shape = (total_len, bs, emb_dim) new_mems[i].shape = (MEM_LEN, bs, d_model)
        hidden = hidden.permute(1,0,2)
        num_mem = self.num_mem_tokens
        
        if self.num_mem_tokens > 0:
            if self.mem_at_end:
                tgt_len = token_embeddings.shape[1]
                mem_tokens_write = hidden[:, -num_mem:, :]
            else:
                tgt_len = token_embeddings.shape[1]
                mem_tokens_write = hidden[:, -tgt_len-num_mem:-tgt_len, :]

            if self.n_head_ca != 0:
                if self.mrv_act is not None:
                    new_mem_tokens = self.mrv_act(hidden[:, -num_mem:, :])
                else:
                    new_mem_tokens = hidden[:, -num_mem:, :]

                mem_tokens = mem_tokens.permute(1,0,2)
                mask_mem_mem = torch.ones((new_mem_tokens.shape[1], new_mem_tokens.shape[1]), dtype=torch.bool).to(new_mem_tokens.device)
                mem_tokens_write, _ = self.mha_mem_to_mem(mem_tokens, new_mem_tokens, new_mem_tokens, attn_mask=mask_mem_mem)

        if self.mem_at_end:
            logits = self.head(hidden)[:, num_mem:-num_mem]
        else:
            tgt_len = token_embeddings.shape[1] # total_len
            logits = self.head(hidden)[:, -tgt_len:] # was tgt_len # logits: torch.Size([64, 301, 4])
        
        if actions is not None:
            logits = logits[:, 1::3, :]
        else:
            logits = logits[:, 1:, :]    
        
        output = {
            'logits': logits,
            'new_mems': new_mems if new_mems is not None else None,
            'mem_tokens': mem_tokens_write.permute(1, 0, 2) if self.num_mem_tokens != 0 else None
        }
        
        return output

######################################################################################    

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, scale=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.scale = scale

    def forward(self, query, key, value, attn_mask=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / query.size(-1)**0.5 if self.scale is None else self.scale
    
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if self.is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        #attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, train=True)
        output = torch.matmul(attn_weight, value)
        return output, attn_weight
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model, num_heads, dropout_p=0.0, is_causal=False, scale=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_p, is_causal, scale)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_q, d_model)
        self.wk = nn.Linear(d_k, d_model)
        self.wv = nn.Linear(d_v, d_model)
        
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)
        # q: (bs, sl, d_q)
        # k: (bs, sl, d_k)
        # q: (bs, sel, d_v)
        
        q = self.wq(q) # (bs, sl, d_model)
        k = self.wk(k) # (bs, sl, d_model)
        v = self.wv(v) # (bs, sl, d_model)
        
        q = self.split_heads(q, batch_size) # (bs, num_heads, seq_len, depth=d_model/num_heads)
        k = self.split_heads(k, batch_size) # (bs, num_heads, seq_len, depth=d_model/num_heads)
        v = self.split_heads(v, batch_size) # (bs, num_heads, seq_len, depth=d_model/num_heads)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, attn_mask)
        
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        
        # (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        
        return output, attention_weights