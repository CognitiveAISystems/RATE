import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from RATE.env_encoders import ObsEncoder, ActDecoder

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

class ImplicitQLearning(nn.Module):
    def __init__(
            self, 
            state_dim, 
            act_dim,
            d_model=128, 
            hidden_layers=[128, 128],
            dropout=0.1,
            env_name='mujoco',
            padding_idx=None,
            tau=0.7,
            beta=1.0,
            discount=0.99,
            alpha=0.01,
            **kwargs
        ):
        
        super(ImplicitQLearning, self).__init__()

        self.d_embed = d_model
        self.d_model = d_model
        self.env_name = env_name
        self.act_dim = act_dim
        self.padding_idx = padding_idx
        self.mem_tokens = None
        self.attn_map = None
        
        # IQL specific parameters
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        
        # State encoder (shared between policy, Q, and V networks)
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        
        # Policy network (similar to BC)
        policy_layers = []
        input_dim = d_model
        for hidden_dim in hidden_layers:
            policy_layers.append(nn.Linear(input_dim, hidden_dim))
            policy_layers.append(nn.ReLU())
            if dropout > 0:
                policy_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
            
        self.policy_mlp = nn.Sequential(*policy_layers)
        self.policy_head = ActDecoder(self.env_name, act_dim, hidden_layers[-1]).act_decoder
        
        # Q-function (twin Q networks)
        self.q1_mlp = self._build_mlp(d_model + act_dim, hidden_layers, dropout)
        self.q1_head = nn.Linear(hidden_layers[-1], 1)
        
        self.q2_mlp = self._build_mlp(d_model + act_dim, hidden_layers, dropout)
        self.q2_head = nn.Linear(hidden_layers[-1], 1)
        
        # Target Q networks
        self.target_q1_mlp = copy.deepcopy(self.q1_mlp)
        self.target_q1_head = copy.deepcopy(self.q1_head)
        self.target_q2_mlp = copy.deepcopy(self.q2_mlp)
        self.target_q2_head = copy.deepcopy(self.q2_head)
        
        # Value function
        self.v_mlp = self._build_mlp(d_model, hidden_layers, dropout)
        self.v_head = nn.Linear(hidden_layers[-1], 1)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy_mlp.parameters()) + list(self.policy_head.parameters()), 
            lr=3e-4
        )
        
        self.q_optimizer = torch.optim.Adam(
            list(self.q1_mlp.parameters()) + list(self.q1_head.parameters()) +
            list(self.q2_mlp.parameters()) + list(self.q2_head.parameters()),
            lr=3e-4
        )
        
        self.v_optimizer = torch.optim.Adam(
            list(self.v_mlp.parameters()) + list(self.v_head.parameters()),
            lr=3e-4
        )

    def _build_mlp(self, input_dim, hidden_layers, dropout):
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        return nn.Sequential(*layers)
    
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
    
    def encode_states(self, states):
        B, B1, states, reshape_required = self.reshape_states(states)
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        return state_embeddings
    
    def q_function(self, state_embeddings, actions, target=False):
        # Concatenate state embeddings and actions
        sa = torch.cat([state_embeddings, actions], dim=-1)
        
        # Reshape for processing through MLP
        batch_shape = sa.shape[:-1]
        sa_flat = sa.reshape(-1, sa.shape[-1])
        
        # Process through the appropriate Q network
        if not target:
            q1_features = self.q1_mlp(sa_flat)
            q1_values = self.q1_head(q1_features).reshape(*batch_shape, 1)
            
            q2_features = self.q2_mlp(sa_flat)
            q2_values = self.q2_head(q2_features).reshape(*batch_shape, 1)
            
            return q1_values, q2_values
        else:
            q1_features = self.target_q1_mlp(sa_flat)
            q1_values = self.target_q1_head(q1_features).reshape(*batch_shape, 1)
            
            q2_features = self.target_q2_mlp(sa_flat)
            q2_values = self.target_q2_head(q2_features).reshape(*batch_shape, 1)
            
            return q1_values, q2_values
    
    def v_function(self, state_embeddings):
        # Reshape for processing through MLP
        batch_shape = state_embeddings.shape[:-1]
        s_flat = state_embeddings.reshape(-1, state_embeddings.shape[-1])
        
        # Process through value network
        v_features = self.v_mlp(s_flat)
        v_values = self.v_head(v_features).reshape(*batch_shape, 1)
        
        return v_values
    
    def policy(self, state_embeddings):
        # Reshape for processing through MLP
        batch_shape = state_embeddings.shape[:-1]
        s_flat = state_embeddings.reshape(-1, state_embeddings.shape[-1])
        
        # Process through policy network
        policy_features = self.policy_mlp(s_flat)
        policy_actions = self.policy_head(policy_features)
        
        # Reshape back to original batch dimensions
        policy_actions = policy_actions.reshape(*batch_shape, self.act_dim)
        
        return policy_actions
    
    def forward(self, states, actions=None, rtgs=None, target=None, timesteps=None, *args, **kwargs):
        # Encode states
        state_embeddings = self.encode_states(states)
        
        # Get policy actions
        policy_actions = self.policy(state_embeddings)
        
        output = {
            'logits': policy_actions,  # For compatibility with BC interface
            'new_mems': None,
            'mem_tokens': None
        }
        
        return output
    
    def update(self, observations, actions, next_observations, rewards, terminals):
        """
        Perform IQL update on batched data
        
        Args:
            observations: (B, T, obs_dim) tensor of observations
            actions: (B, T, act_dim) tensor of actions
            next_observations: (B, T, obs_dim) tensor of next observations
            rewards: (B, T) tensor of rewards
            terminals: (B, T) tensor of terminals (1 if terminal, 0 otherwise)
        """
        # Encode states
        state_embeddings = self.encode_states(observations)
        next_state_embeddings = self.encode_states(next_observations)
        
        # Compute target Q values
        with torch.no_grad():
            target_q1, target_q2 = self.q_function(state_embeddings, actions, target=True)
            target_q = torch.min(target_q1, target_q2)
            next_v = self.v_function(next_state_embeddings)
        
        # Update value function
        v = self.v_function(state_embeddings)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        
        self.v_optimizer.zero_grad()
        v_loss.backward(retain_graph=True)  # Add retain_graph=True here
        self.v_optimizer.step()
        
        # Update Q function
        # Reshape rewards and terminals to match next_v
        rewards = rewards.unsqueeze(-1)  # (B, T, 1)
        terminals = terminals.unsqueeze(-1)  # (B, T, 1)
        
        targets = rewards + (1.0 - terminals) * self.discount * next_v.detach()
        q1, q2 = self.q_function(state_embeddings, actions)
        q1_loss = F.mse_loss(q1, targets)
        q2_loss = F.mse_loss(q2, targets)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward(retain_graph=True)  # Add retain_graph=True here
        self.q_optimizer.step()
        
        # Update target Q networks
        update_exponential_moving_average(self.target_q1_mlp, self.q1_mlp, self.alpha)
        update_exponential_moving_average(self.target_q1_head, self.q1_head, self.alpha)
        update_exponential_moving_average(self.target_q2_mlp, self.q2_mlp, self.alpha)
        update_exponential_moving_average(self.target_q2_head, self.q2_head, self.alpha)
        
        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100.0)
        policy_actions = self.policy(state_embeddings)
        bc_losses = torch.sum((policy_actions - actions)**2, dim=-1, keepdim=True)
        policy_loss = torch.mean(exp_adv * bc_losses)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()  # No need for retain_graph=True on the last backward
        self.policy_optimizer.step()
        
        # Return losses for logging
        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'adv_mean': adv.mean().item(),
            'exp_adv_mean': exp_adv.mean().item()
        }