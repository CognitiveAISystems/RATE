import torch
import torch.nn as nn
import torch.nn.functional as F

from RATE.env_encoders import ObsEncoder, ActDecoder

class BehaviorCloning(nn.Module):
    def __init__(self, 
                 state_dim, 
                 act_dim,
                 d_model=25, 
                 hidden_layers=[256, 256],
                 dropout=0.1,
                 env_name='mujoco',
                 padding_idx=None,
                 **kwargs
                 ):
        
        super(BehaviorCloning, self).__init__()

        self.d_embed = d_model
        self.d_model = d_model
        self.env_name = env_name
        self.act_dim = act_dim
        self.padding_idx = padding_idx
        self.mem_tokens = None
        self.attn_map = None
        
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        
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
        
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        

        features = self.mlp(state_embeddings)
        logits = self.head(features)

        
        output = {
            'logits': logits,
            'new_mems': None,
            'mem_tokens': None
        }
        
        return output