import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .normalization import get_norm_layer


class SwiGLU(nn.Module):
    """SwiGLU activation function - a gated linear unit using Swish/SiLU.

    Takes input x and applies: SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    More efficient than GELU while maintaining similar performance.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Gate projection
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        # Value projection  
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        # Down projection
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(gate) * up
        gate = F.silu(self.gate_proj(x))  # Swish/SiLU activation
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class Expert(nn.Module):
    """Single expert in MoE - uses SwiGLU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class Router(nn.Module):
    """Router network for MoE that selects which experts to use"""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts based on learned probabilities.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            top_k: Override default top_k (handy for reserving slots for shared experts)

        Returns:
            gates: Routing weights for selected experts [batch, seq_len, top_k]
            indices: Expert indices for each token [batch, seq_len, top_k]
        """
        tk = self.top_k if top_k is None else top_k
        router_logits = self.linear(x)                       # [B, S, E]
        router_probs  = F.softmax(router_logits, dim=-1)
        gates, indices = torch.topk(router_probs, tk, dim=-1)   # [B, S, tk]
        gates = gates / gates.sum(dim=-1, keepdim=True)
        return gates, indices


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        # Legacy d_ff parameter - kept for compatibility
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,                     # Total experts per token (routed + shared)
        dropout: float = 0.1,
        expert_dropout: float = 0.1,
        load_balancing_loss_coef: float = 0.01,
        # Inspired by DeepSeek - using shared experts
        use_shared_expert: bool = True,
        n_shared_experts: int = 1,
        shared_d_ff: Optional[int] = None,
        routed_d_ff: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.n_shared = n_shared_experts if use_shared_expert else 0
        self.top_k_total = top_k
        self.top_k_routed = max(1, top_k - self.n_shared)  # Reserve slots for shared experts
        self.load_balancing_loss_coef = load_balancing_loss_coef

        # Router over *routed* experts only
        self.router = Router(d_model, num_experts, top_k=self.top_k_routed)

        # Set feed-forward dimensions
        routed_width = routed_d_ff or d_ff
        shared_width = shared_d_ff or d_ff

        # Create the routed experts
        self.experts = nn.ModuleList([
            Expert(d_model, routed_width, expert_dropout)
            for _ in range(num_experts)
        ])

        # Shared expert(s): always-on; sum all shared branches
        if self.n_shared > 0:
            self.shared_experts = nn.ModuleList([
                Expert(d_model, shared_width, expert_dropout)
                for _ in range(self.n_shared)
            ])
        else:
            self.shared_experts = nn.ModuleList()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape

        # Route tokens through the selected experts
        gates, indices = self.router(x, top_k=self.top_k_routed)  # [B,S,Kr]
        routed_out = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            mask = (indices == expert_idx)                 # [B,S,Kr]
            if mask.any():
                g = (gates * mask.float()).sum(dim=-1, keepdim=True)  # [B,S,1]
                tokens = g.squeeze(-1) > 0
                if tokens.any():
                    xin = x[tokens]                       # [N,D]
                    xout = self.experts[expert_idx](xin)  # [N,D]
                    contrib = torch.zeros_like(x)
                    contrib[tokens] = xout
                    routed_out += contrib * g

        # Process through shared experts (always active)
        shared_out = torch.zeros_like(x)
        for se in self.shared_experts:
            shared_out += se(x.reshape(-1, D)).reshape(B, S, D)

        y = self.dropout(routed_out + shared_out)

        # load-balancing across routed only
        aux = None
        if self.training and self.load_balancing_loss_coef > 0:
            aux = self._calculate_load_balancing_loss(gates, indices)
        return y, aux
    
    def _calculate_load_balancing_loss(self, gates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Calculate load balancing loss to encourage equal expert usage"""
        batch_size, seq_len, top_k = gates.shape
        
        # Count how many tokens are routed to each expert
        expert_counts = torch.zeros(self.num_experts, device=gates.device, dtype=gates.dtype)
        for expert_idx in range(self.num_experts):
            expert_mask = (indices == expert_idx).float()
            expert_counts[expert_idx] = expert_mask.sum()
        
        # Calculate the fraction of tokens routed to each expert
        total_tokens = batch_size * seq_len * top_k
        expert_fractions = expert_counts / total_tokens
        
        # Calculate the average gate value for each expert
        gate_sums = torch.zeros(self.num_experts, device=gates.device, dtype=gates.dtype)
        for expert_idx in range(self.num_experts):
            expert_mask = (indices == expert_idx).float()
            gate_sums[expert_idx] = (gates * expert_mask).sum()
        
        gate_averages = gate_sums / total_tokens
        
        # Load balancing loss: encourage uniform distribution
        load_balancing_loss = (expert_fractions * gate_averages).sum() * self.num_experts
        
        return self.load_balancing_loss_coef * load_balancing_loss


class MoEFeedForwardNetwork(nn.Module):
    def __init__(self,
        d_model: int, d_ff: int, dropout: float = 0.1, pre_lnorm: bool = True,
        norm_type: Optional[str] = None, use_moe: bool = False,
        num_experts: int = 8, top_k: int = 2,
        expert_dropout: float = 0.1, load_balancing_loss_coef: float = 0.01,
        use_swiglu: bool = True,
        # Additional options for expert configuration
        use_shared_expert: bool = True,
        n_shared_experts: int = 1,
        shared_d_ff: Optional[int] = None,
        routed_d_ff: Optional[int] = None
    ):
        super().__init__()
        self.pre_lnorm = pre_lnorm
        self.use_moe = use_moe
        self.use_swiglu = use_swiglu
        self.layer_norm = get_norm_layer(norm_type, d_model)

        if use_moe:
            self.core_net = MixtureOfExperts(
                d_model=d_model,
                d_ff=d_ff,  # Legacy parameter for compatibility
                num_experts=num_experts,
                top_k=top_k,  # Total experts per token
                dropout=dropout,
                expert_dropout=expert_dropout,
                load_balancing_loss_coef=load_balancing_loss_coef,
                use_shared_expert=use_shared_expert,
                n_shared_experts=n_shared_experts,
                shared_d_ff=shared_d_ff,
                routed_d_ff=routed_d_ff,
            )
        else:
            # Use standard FFN with optional SwiGLU
            if use_swiglu:
                self.core_net = SwiGLU(d_model, d_ff, dropout)
            else:
                # Original GELU-based FFN
                self.core_net = nn.Sequential(
                    nn.Linear(d_model, d_ff), 
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional layer normalization and MoE.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            output: Processed tensor [batch_size, seq_len, d_model]
            aux_loss: Load balancing loss if using MoE (None otherwise)
        """
        aux_loss = None
        
        if self.pre_lnorm:
            # Pre-layer normalization
            normalized_x = self.layer_norm(x)
            
            if self.use_moe:
                core_out, aux_loss = self.core_net(normalized_x)
            else:
                if hasattr(self.core_net, '__call__') and not isinstance(self.core_net, nn.Sequential):
                    # SwiGLU case
                    core_out = self.core_net(normalized_x)
                else:
                    # Sequential case
                    core_out = self.core_net(normalized_x)
            
            # Residual connection
            output = core_out + x
        else:
            # Post-layer normalization
            if self.use_moe:
                core_out, aux_loss = self.core_net(x)
            else:
                if hasattr(self.core_net, '__call__') and not isinstance(self.core_net, nn.Sequential):
                    # SwiGLU case
                    core_out = self.core_net(x)
                else:
                    # Sequential case
                    core_out = self.core_net(x)
            
            # Residual connection + layer normalization
            output = self.layer_norm(x + core_out)
        
        return output, aux_loss
    
    def get_expert_usage_stats(self) -> Optional[dict]:
        """Get statistics about expert usage (only for MoE)"""
        if not self.use_moe:
            return None
            
        # TODO: Add proper expert usage tracking
        # Currently returns basic structural info
        return {
            "num_experts": self.core_net.num_experts,
            "top_k": self.core_net.top_k,
            "total_parameters": sum(p.numel() for p in self.core_net.parameters()),
            "expert_parameters": sum(p.numel() for expert in self.core_net.experts for p in expert.parameters())
        }
