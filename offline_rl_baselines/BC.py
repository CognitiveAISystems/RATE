import torch
import torch.nn as nn
import torch.nn.functional as F

from RATE.env_encoders import ObsEncoder, ActDecoder, ActEncoder


class BehaviorCloning(nn.Module):
    def __init__(
        self, 
        state_dim, 
        act_dim,
        d_model=64, 
        hidden_layers=[128, 128],
        dropout=0.1,
        env_name='mujoco',
        padding_idx=None,
        backbone='mlp',  # Choice of architecture: 'mlp' or 'lstm'
        lstm_layers=1,   # Parameters for LSTM
        bidirectional=False,
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
        self.backbone = backbone.lower()
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        self.action_embeddings  = ActEncoder(self.env_name, act_dim, self.d_embed).act_encoder
        
        # Choosing backbone architecture
        if self.backbone == 'mlp':
            layers = []
            input_dim = d_model
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            self.backbone_net = nn.Sequential(*layers)
            self.output_dim = hidden_layers[-1]
            
        elif self.backbone == 'lstm':
            self.hidden_size = hidden_layers[-1]
            self.backbone_net = nn.LSTM(
                input_size=d_model,
                hidden_size=self.hidden_size,
                num_layers=lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
            self.output_dim = self.hidden_size * self.num_directions
            
            # Weight initialization for LSTM
            for name, param in self.backbone_net.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone}. Choose 'mlp' or 'lstm'")
        
        self.head = ActDecoder(self.env_name, act_dim, self.output_dim).act_decoder

    def init_hidden(self, batch_size, device):
        """Weight initialization for LSTM"""
        # Dimensions: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(
            self.lstm_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.lstm_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        
        # Weight initialization using orthogonal matrix for better convergence
        nn.init.orthogonal_(h_0)
        nn.init.orthogonal_(c_0)
        
        return h_0, c_0

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
    
    def forward(self, states, actions=None, rtgs=None, target=None, timesteps=None, hidden=None, *args, **kwargs):
        B, B1, states, reshape_required = self.reshape_states(states)
        
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        
        # Different processing for different backbones
        if self.backbone == 'mlp':
            features = self.backbone_net(state_embeddings)
            new_hidden = None
        else:  # lstm
            # If hidden states are not passed or this is the start of a new sequence,
            # initialize new ones
            if hidden is None:
                hidden = self.init_hidden(B, states.device)
            
            # Apply LSTM
            features, new_hidden = self.backbone_net(state_embeddings, hidden)
            
        logits = self.head(features)
        
        output = {
            'logits': logits,
            'new_mems': None,
            'mem_tokens': None,
            'hidden': new_hidden  # Save new hidden states
        }
        
        return output

    def reset_hidden(self, batch_size=None, device=None):
        """Reset LSTM hidden states"""
        if self.backbone == 'lstm':
            if batch_size is None or device is None:
                return None
            return self.init_hidden(batch_size, device)
        return None