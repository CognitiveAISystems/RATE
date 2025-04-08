import torch
import torch.nn as nn
import torch.nn.functional as F

from RATE.env_encoders import ObsEncoder, ActDecoder

class ConservativeQLearning(nn.Module):
    def __init__(self, 
                 state_dim, 
                 act_dim,
                 d_model=25, 
                 hidden_layers=[256, 256],
                 dropout=0.1,
                 env_name='mujoco',
                 padding_idx=None,
                 cql_alpha=1.0,
                 **kwargs
                 ):
        
        super(ConservativeQLearning, self).__init__()

        self.d_embed = d_model
        self.d_model = d_model
        self.env_name = env_name
        self.act_dim = act_dim
        self.padding_idx = padding_idx
        self.mem_tokens = None
        self.attn_map = None
        self.cql_alpha = cql_alpha
        
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        
        # Policy network (BC part)
        layers = []
        input_dim = d_model
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        
        self.head = ActDecoder(self.env_name, act_dim, hidden_layers[-1]).act_decoder
        
        # Q-networks for CQL
        self.q1 = nn.Sequential(
            nn.Linear(self.d_embed + act_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(self.d_embed + act_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], 1)
        )

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
    
    def forward(self, states, actions=None, rtgs=None, target=None, timesteps=None, *args, **kwargs):
        B, B1, states, reshape_required = self.reshape_states(states)
        
        # Encode states
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        
        # Policy network (BC part)
        features = self.mlp(state_embeddings)
        logits = self.head(features)
        
        # CQL part
        cql_loss = None
        q1_value = None
        q2_value = None
        
        if actions is not None:
            # For CQL, we need to flatten the batch and sequence dimensions
            # to process each state-action pair independently
            state_embeddings_flat = state_embeddings.reshape(-1, self.d_embed)
            actions_flat = actions.reshape(-1, self.act_dim)
                
            state_action = torch.cat([state_embeddings_flat, actions_flat], dim=-1)
            q1_value = self.q1(state_action)
            q2_value = self.q2(state_action)
            
            # For continuous actions in mikasa-robo, we sample random actions from a normal distribution
            if self.env_name == 'mikasa_robo':
                # Sample random actions for CQL regularization
                batch_size = state_embeddings_flat.shape[0]
                random_actions = torch.randn(batch_size, self.act_dim).to(state_embeddings.device)
                random_actions = torch.clamp(random_actions, -1.0, 1.0)  # Assuming actions are normalized to [-1, 1]
                
                # Get Q-values for random actions
                random_state_action = torch.cat([state_embeddings_flat, random_actions], dim=-1)
                random_q1 = self.q1(random_state_action)
                random_q2 = self.q2(random_state_action)
                
                # CQL loss calculation for continuous actions
                cql_loss = (torch.logsumexp(random_q1, dim=0) - q1_value.mean()).mean() + \
                           (torch.logsumexp(random_q2, dim=0) - q2_value.mean()).mean()
                cql_loss = self.cql_alpha * cql_loss
            else:
                # For discrete actions, sample uniformly from action space
                batch_size = state_embeddings_flat.shape[0]
                random_actions = torch.randint(0, self.act_dim, (batch_size, 1)).to(state_embeddings.device)
                random_actions_one_hot = F.one_hot(random_actions.squeeze(-1), num_classes=self.act_dim).float()
                
                random_state_action = torch.cat([state_embeddings_flat, random_actions_one_hot], dim=-1)
                random_q1 = self.q1(random_state_action)
                random_q2 = self.q2(random_state_action)
                
                cql_loss = (torch.logsumexp(random_q1, dim=0) - q1_value.mean()).mean() + \
                           (torch.logsumexp(random_q2, dim=0) - q2_value.mean()).mean()
                cql_loss = self.cql_alpha * cql_loss
        
        # Reshape Q-values back to batch dimensions if needed
        if q1_value is not None and reshape_required:
            q1_value = q1_value.reshape(B, B1, 1)
            q2_value = q2_value.reshape(B, B1, 1)
        
        output = {
            'logits': logits,
            'new_mems': None,
            'mem_tokens': None,
            'q1_value': q1_value,
            'q2_value': q2_value,
            'cql_loss': cql_loss
        }
        
        return output