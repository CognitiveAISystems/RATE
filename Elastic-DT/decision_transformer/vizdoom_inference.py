"""Inference utilities for Elastic-DT on ViZDoom."""

import torch
import numpy as np


class ViZDoomEDTWrapper:
    """Wrapper to make Elastic-DT compatible with RATE validation interface."""
    
    def __init__(self, edt_model):
        """
        Args:
            edt_model: ElasticDecisionTransformerViZDoom model
        """
        self.edt_model = edt_model
        self.backbone = None  # Not LSTM
        self.mem_tokens = None
        
    def __call__(self, states, actions, rtgs, rewards, timesteps, **kwargs):
        """
        Forward pass compatible with RATE validation interface.
        
        Args:
            states: (B, T, C, H, W) - image observations
            actions: (B, T) - discrete action indices (can be None for first step)
            rtgs: (B, T, 1) - returns-to-go
            rewards: Not used in Elastic-DT
            timesteps: (B, T) - timestep indices
            
        Returns:
            dict with 'logits' key containing action logits
        """
        # Ensure timesteps are 2D and get dimensions
        if len(timesteps.shape) == 3:
            timesteps = timesteps.squeeze(-1)
        
        # Ensure rtgs are 3D (B, T, 1)
        if len(rtgs.shape) == 2:
            rtgs = rtgs.unsqueeze(-1)
        
        # Get the actual sequence length from states (most reliable for context window)
        B, T_states = states.shape[0], states.shape[1]
        
        # Truncate all inputs to the same length (use states length as reference)
        T = T_states
        timesteps = timesteps[:, :T]
        rtgs = rtgs[:, :T, :]
        
        # Fix state dimensions: ViZDoom returns (C, W, H) = (3, 112, 64)
        # but model expects (C, H, W) = (3, 64, 112)
        # Input states: (B, T, C, W, H) -> Need: (B, T, C, H, W)
        if states.shape[-2:] == (112, 64):  # Check if dimensions need swapping
            states = states.transpose(-2, -1)  # Swap W and H
        
        # Handle None actions (first timestep)
        if actions is None:
            # Create dummy actions with correct length T from states
            actions = torch.zeros((B, T), dtype=torch.long, device=states.device)
        else:
            # Ensure actions are 2D (B, T-1) or (B, T)
            # Actions come from validation as (B, T-1, 1) after [:, 1:, :] slicing
            if len(actions.shape) == 3:
                actions = actions.squeeze(-1)
            # Ensure long type for embedding lookup
            actions = actions.long()
            
            # Pad actions to match sequence length T
            # Validation sends actions with length T-1 (skips first action)
            if actions.shape[1] < T:
                # Prepend a zero action (dummy for first timestep)
                padding = torch.zeros((B, T - actions.shape[1]), 
                                    dtype=torch.long, device=actions.device)
                actions = torch.cat([padding, actions], dim=1)
            elif actions.shape[1] > T:
                # Truncate if somehow longer
                actions = actions[:, :T]
        
        # Forward pass
        _, action_preds, _, _, _ = self.edt_model(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=rtgs,
            rewards=None
        )
        
        return {
            'logits': action_preds,
            'new_mems': None,
            'mem_tokens': None,
            'hidden': None,
            'memory_states': None
        }
    
    def eval(self):
        """Set model to evaluation mode."""
        self.edt_model.eval()
    
    def train(self):
        """Set model to training mode."""
        self.edt_model.train()
    
    def to(self, device):
        """Move model to device."""
        self.edt_model.to(device)
        return self
    
    def parameters(self):
        """Get model parameters."""
        return self.edt_model.parameters()


@torch.no_grad()
def sample_edt_vizdoom(
    model, states, actions, rtgs, timesteps, context_length, device
):
    """
    Sample action from Elastic-DT model for ViZDoom.
    
    Args:
        model: ElasticDecisionTransformerViZDoom or ViZDoomEDTWrapper
        states: (B, T, C, H, W) - current states
        actions: (B, T) or None - previous actions
        rtgs: (B, T, 1) - returns-to-go
        timesteps: (B, T) - timesteps
        context_length: Maximum context length
        device: torch device
        
    Returns:
        logits: (B, act_dim) - action logits for the last timestep
    """
    model.eval()
    
    B, T = states.shape[0], states.shape[1]
    
    # Crop to context length if needed
    if T > context_length:
        states = states[:, -context_length:]
        if actions is not None:
            actions = actions[:, -context_length:]
        rtgs = rtgs[:, -context_length:]
        timesteps = timesteps[:, -context_length:]
        T = context_length
    
    # Handle None actions
    if actions is None:
        actions = torch.zeros((B, T), dtype=torch.long, device=device)
    
    # Ensure actions are 2D
    if len(actions.shape) == 3:
        actions = actions.squeeze(-1)
    
    # Forward pass
    if isinstance(model, ViZDoomEDTWrapper):
        result = model(states, actions, rtgs, None, timesteps)
        logits = result['logits']
    else:
        _, logits, _, _, _ = model(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=rtgs,
            rewards=None
        )
    
    # Return logits for the last timestep
    return logits[:, -1, :]

