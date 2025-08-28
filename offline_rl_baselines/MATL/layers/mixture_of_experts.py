import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .normalization import get_norm_layer


class SwiGLU(nn.Module):
    """SwiGLU activation function: SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    
    This is a gated linear unit that uses Swish (SiLU) activation.
    It's more parameter-efficient than GELU for the same performance.
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
        """
        If top_k is given, override the default self.top_k (useful when reserving 1 slot for the shared expert).
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
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,                          # total K you pass from config
        dropout: float = 0.1,
        expert_dropout: float = 0.1,
        load_balancing_loss_coef: float = 0.01,
        # --- new ---
        use_shared_expert: bool = True,          # enable shared expert
        shared_gate_mode: str = "learned",       # {"learned","fixed"}
        shared_gate_init: float = 0.0,           # sigmoid(init) ~ 0.5 when 0.0
        shared_alpha_fixed: float = 0.2          # used when shared_gate_mode=="fixed"
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts           # routed experts only
        self.top_k_total = top_k                 # includes shared if enabled
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.use_shared_expert = use_shared_expert
        self.shared_gate_mode = shared_gate_mode
        self.shared_alpha_fixed = shared_alpha_fixed

        # Router over *routed* experts
        self.router = Router(d_model, num_experts, top_k=max(1, top_k - (1 if use_shared_expert else 0)))

        # Routed experts
        # self.experts = nn.ModuleList([Expert(d_model, d_ff, expert_dropout) for _ in range(num_experts)])
        # Routed experts (narrower: d_ff // 4)
        self.experts = nn.ModuleList([
            Expert(d_model, max(1, d_ff // 4), expert_dropout)   # NEW
            for _ in range(num_experts)
        ])

        # Shared expert + isolated gate
        if use_shared_expert:
            self.shared_expert = Expert(d_model, d_ff, expert_dropout)
            if shared_gate_mode == "learned":
                self.shared_gate = nn.Linear(d_model, 1, bias=True)
                with torch.no_grad():
                    self.shared_gate.bias.fill_(shared_gate_init)
                    if hasattr(self.shared_gate, "weight"):
                        nn.init.zeros_(self.shared_gate.weight)
            else:
                self.register_parameter("shared_gate", None)
        else:
            self.register_parameter("shared_gate", None)
            self.shared_expert = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, D]
        returns: y, aux_loss
        """
        B, S, D = x.shape
        # --- routed experts path ---
        routed_top_k = max(1, self.router.top_k)   # already reserves 1 slot if shared is on
        routed_gates, routed_indices = self.router(x, top_k=routed_top_k)  # [B,S,Kr]

        routed_out = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            expert_mask = (routed_indices == expert_idx)            # [B,S,Kr]
            if expert_mask.any():
                # sum gates for this expert (tokens that chose it)
                expert_g = (routed_gates * expert_mask.float()).sum(dim=-1, keepdim=True)   # [B,S,1]
                tokens = expert_g.squeeze(-1) > 0                                             # [B,S]
                if tokens.any():
                    expert_in  = x[tokens]                                                    # [N,D]
                    expert_out = self.experts[expert_idx](expert_in)                          # [N,D]
                    contrib = torch.zeros_like(x)
                    contrib[tokens] = expert_out
                    contrib = contrib * expert_g
                    routed_out += contrib

        # --- shared expert path (isolated) ---
        if self.use_shared_expert:
            # all tokens go through the shared expert
            shared_out = self.shared_expert(x.reshape(-1, D)).reshape(B, S, D)                # [B,S,D]
            if self.shared_gate_mode == "learned":
                alpha = torch.sigmoid(self.shared_gate(x))                                    # [B,S,1]
            else:
                alpha = torch.full((B, S, 1), self.shared_alpha_fixed, device=x.device, dtype=x.dtype)
            y = alpha * shared_out + (1.0 - alpha) * routed_out
        else:
            y = routed_out

        y = self.dropout(y)

        # load-balancing only across *routed* experts (isolation)
        aux = None
        if self.training and self.load_balancing_loss_coef > 0:
            aux = self._calculate_load_balancing_loss(routed_gates, routed_indices)
        return y, aux
    
    def _calculate_load_balancing_loss(self, gates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Calculate load balancing loss to encourage equal expert usage"""
        batch_size, seq_len, top_k = gates.shape
        
        # Count how many tokens are routed to each expert
        expert_counts = torch.zeros(self.num_experts, device=gates.device)
        for expert_idx in range(self.num_experts):
            expert_mask = (indices == expert_idx).float()
            expert_counts[expert_idx] = expert_mask.sum()
        
        # Calculate the fraction of tokens routed to each expert
        total_tokens = batch_size * seq_len * top_k
        expert_fractions = expert_counts / total_tokens
        
        # Calculate the average gate value for each expert
        gate_sums = torch.zeros(self.num_experts, device=gates.device)
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
        # --- new (all optional) ---
        use_shared_expert: bool = True,
        shared_gate_mode: str = "learned",
        shared_gate_init: float = 0.0,
        shared_alpha_fixed: float = 0.2
    ):
        super().__init__()
        self.pre_lnorm = pre_lnorm
        self.use_moe = use_moe
        self.use_swiglu = use_swiglu
        self.layer_norm = get_norm_layer(norm_type, d_model)

        if use_moe:
            self.core_net = MixtureOfExperts(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                expert_dropout=expert_dropout,
                load_balancing_loss_coef=load_balancing_loss_coef,
                use_shared_expert=use_shared_expert,
                shared_gate_mode=shared_gate_mode,
                shared_gate_init=shared_gate_init,
                shared_alpha_fixed=shared_alpha_fixed,
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
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            aux_loss: Optional auxiliary loss (load balancing loss for MoE)
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
            
        # This would need to be implemented with proper tracking
        # For now, return basic info
        return {
            "num_experts": self.core_net.num_experts,
            "top_k": self.core_net.top_k,
            "total_parameters": sum(p.numel() for p in self.core_net.parameters()),
            "expert_parameters": sum(p.numel() for expert in self.core_net.experts for p in expert.parameters())
        }
