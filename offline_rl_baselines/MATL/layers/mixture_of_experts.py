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
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            gates: [batch_size, seq_len, top_k] - routing weights
            indices: [batch_size, seq_len, top_k] - expert indices
        """
        # Compute routing logits
        router_logits = self.linear(x)  # [batch_size, seq_len, num_experts]
        
        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        gates, indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Renormalize the selected gates so they sum to 1
        gates = gates / gates.sum(dim=-1, keepdim=True)
        
        return gates, indices


class MixtureOfExperts(nn.Module):
    """Mixture of Experts module with SwiGLU experts"""
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        num_experts: int = 8, 
        top_k: int = 2,
        dropout: float = 0.1,
        expert_dropout: float = 0.1,
        load_balancing_loss_coef: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_loss_coef = load_balancing_loss_coef
        
        # Router
        self.router = Router(d_model, num_experts, top_k)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, expert_dropout) 
            for _ in range(num_experts)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            load_balancing_loss: Optional load balancing loss for training
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get routing decisions
        gates, indices = self.router(x)  # gates: [B, S, top_k], indices: [B, S, top_k]
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx)  # [B, S, top_k]
            
            if expert_mask.any():
                # Get the corresponding gates for this expert
                expert_gates = gates * expert_mask.float()  # [B, S, top_k]
                expert_gates = expert_gates.sum(dim=-1, keepdim=True)  # [B, S, 1]
                
                # Only process tokens that use this expert
                tokens_for_expert = expert_gates.squeeze(-1) > 0  # [B, S]
                
                if tokens_for_expert.any():
                    # Extract tokens for this expert
                    expert_input = x[tokens_for_expert]  # [num_tokens, d_model]
                    
                    # Process through expert
                    expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, d_model]
                    
                    # Add weighted expert output back
                    expert_contribution = torch.zeros_like(x)
                    expert_contribution[tokens_for_expert] = expert_output
                    expert_contribution = expert_contribution * expert_gates
                    
                    output += expert_contribution
        
        output = self.dropout(output)
        
        # Calculate load balancing loss for training
        load_balancing_loss = None
        if self.training and self.load_balancing_loss_coef > 0:
            load_balancing_loss = self._calculate_load_balancing_loss(gates, indices)
        
        return output, load_balancing_loss
    
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
    """MoE-enhanced Feed Forward Network with option to use standard FFN or MoE"""
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout: float = 0.1, 
        pre_lnorm: bool = True, 
        norm_type: Optional[str] = None,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        expert_dropout: float = 0.1,
        load_balancing_loss_coef: float = 0.01,
        use_swiglu: bool = True
    ):
        super().__init__()
        self.pre_lnorm = pre_lnorm
        self.use_moe = use_moe
        self.use_swiglu = use_swiglu
        
        self.layer_norm = get_norm_layer(norm_type, d_model)
        
        if use_moe:
            # Use Mixture of Experts
            self.core_net = MixtureOfExperts(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                expert_dropout=expert_dropout,
                load_balancing_loss_coef=load_balancing_loss_coef
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
