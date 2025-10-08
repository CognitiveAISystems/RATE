import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import get_norm_layer


class RelPartialLearnableMultiHeadAttn(nn.Module):
    """
    Relative Partial Learnable Multi-Head Attention from Transformer-XL
    """
    
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, dropatt: float = 0.0, pre_lnorm: bool = True, norm_type=None):
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.pre_lnorm = pre_lnorm
        
        self.qkv_net = nn.Linear(d_model, 3 * n_head * self.d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head * self.d_head, bias=False)
        
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * self.d_head, d_model, bias=False)
        
        self.layer_norm = get_norm_layer(norm_type, d_model)
        self.scale = 1 / (self.d_head ** 0.5)
        
    def _rel_shift(self, x, zero_triu=False):
        """Shift relative positions from Transformer-XL"""
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        
        x = x_padded[1:].view_as(x)
        
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]
        
        return x
    
    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        """
        Args:
            w: Input embeddings [qlen, bsz, d_model]
            r: Position embeddings [rlen, 1, d_model] 
            r_w_bias: Content bias [n_head, d_head]
            r_r_bias: Position bias [n_head, d_head]
            attn_mask: Attention mask [qlen, klen, 1] - True values will be masked
            mems: Memory (not used in ELMUR)
        """
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        
        # Apply layer norm depending on pre_lnorm mode
        if self.pre_lnorm:
            w_heads = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)
        r_head_k = self.r_net(r)
        
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        
        klen = w_head_k.size(0)
        
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head
        
        #### Compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        
        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)
        
        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        
        #### Compute attention probability
        if attn_mask is not None and attn_mask.any():
            if attn_mask.dim() == 2:
                # EXACT logic from mem_transformer.py: [None, :, :, None] expansion
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                # For 3D masks (our format: [qlen, klen, 1])
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)
        
        # [qlen x klen x bsz x n_head]
        # attn_score: [qlen, klen, bsz, n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        
        #### Compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        
        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        
        ##### Linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        
        if self.pre_lnorm:
            ##### Residual connection
            output = w + attn_out
        else:
            ##### Residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
        
        return output