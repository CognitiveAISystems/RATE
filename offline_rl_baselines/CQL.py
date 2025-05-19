import torch
import torch.nn as nn
import torch.nn.functional as F

from RATE.env_encoders import ObsEncoder, ActDecoder

class ConservativeQLearning(nn.Module):
    def __init__(
        self, 
        state_dim, 
        act_dim,
        d_model=25, 
        hidden_layers=[256, 256],
        dropout=0.1,
        env_name='mujoco',
        padding_idx=None,
        cql_alpha=1.0,
        backbone='mlp',  # Choice of architecture: 'mlp' or 'lstm'
        lstm_layers=1,   # Parameters for LSTM
        bidirectional=False,
        **kwargs
    ):    
        
        super(ConservativeQLearning, self).__init__()

        self.d_embed = d_model
        self.d_model = d_model
        self.env_name = env_name
        self.act_dim = act_dim
        self.padding_idx = padding_idx
        self.cql_alpha = cql_alpha
        self.mem_tokens = None
        self.attn_map = None
        self.backbone = backbone.lower()
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        
        # Choice of backbone architecture
        if self.backbone == 'mlp':
            # Q1 Network
            q1_layers = []
            input_dim = d_model
            for hidden_dim in hidden_layers:
                q1_layers.append(nn.Linear(input_dim, hidden_dim))
                q1_layers.append(nn.ReLU())
                if dropout > 0:
                    q1_layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            self.q1_backbone = nn.Sequential(*q1_layers)
            
            # Q2 Network
            q2_layers = []
            input_dim = d_model
            for hidden_dim in hidden_layers:
                q2_layers.append(nn.Linear(input_dim, hidden_dim))
                q2_layers.append(nn.ReLU())
                if dropout > 0:
                    q2_layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            self.q2_backbone = nn.Sequential(*q2_layers)
            self.output_dim = hidden_layers[-1]
            
        elif self.backbone == 'lstm':
            self.hidden_size = hidden_layers[-1]
            
            # Use one main LSTM as the main backbone, as in BC
            self.backbone_net = nn.LSTM(
                input_size=d_model,
                hidden_size=self.hidden_size,
                num_layers=lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
            
            # Initialize LSTM weights
            for name, param in self.backbone_net.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            
            # Additional layers for Q1 and Q2 after LSTM
            self.q1_projector = nn.Sequential(
                nn.Linear(self.hidden_size * self.num_directions, self.hidden_size),
                nn.ReLU()
            )
            
            self.q2_projector = nn.Sequential(
                nn.Linear(self.hidden_size * self.num_directions, self.hidden_size),
                nn.ReLU()
            )
            
            self.output_dim = self.hidden_size
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone}. Choose 'mlp' or 'lstm'")
        
        # Q-value heads
        self.q1_head = ActDecoder(self.env_name, act_dim, self.output_dim).act_decoder
        self.q2_head = ActDecoder(self.env_name, act_dim, self.output_dim).act_decoder

    def init_hidden(self, batch_size, device):
        """Initialize LSTM hidden states - in the same format as BC"""
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
        
        # Initialize using orthogonal matrix for better convergence
        nn.init.orthogonal_(h_0)
        nn.init.orthogonal_(c_0)
        
        return h_0, c_0

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
    
    def forward(self, states, actions=None, rtgs=None, target=None, timesteps=None, hidden=None, *args, **kwargs):
        B, B1, states, reshape_required = self.reshape_states(states)
        
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        
        # Different processing for different backbones
        if self.backbone == 'mlp':
            q1_features = self.q1_backbone(state_embeddings)
            q2_features = self.q2_backbone(state_embeddings)
            new_hidden = None
        else:  # lstm
            # If hidden states are not passed or this is the start of a new sequence,
            # initialize new ones
            if hidden is None:
                hidden = self.init_hidden(B, states.device)
            
            # Apply LSTM (one common backbone as in BC)
            features, new_hidden = self.backbone_net(state_embeddings, hidden)
            
            # Project common features to separate Q-networks
            q1_features = self.q1_projector(features)
            q2_features = self.q2_projector(features)
        
        q1_value = self.q1_head(q1_features)
        q2_value = self.q2_head(q2_features)
        
        # CQL logic: compute CQL losses
        cql_loss = None
        
        # Training mode - compute CQL Loss
        if actions is not None and target is not None:
            try:
                # For T-Maze special action handling
                if self.env_name == 'tmaze':
                    # Create a mask of valid values (not padding)
                    if self.padding_idx is not None:
                        if len(actions.shape) == 3 and actions.shape[-1] == 1:
                            mask = (actions.squeeze(-1) != self.padding_idx).float().unsqueeze(-1)
                        else:
                            mask = torch.ones((B, B1, 1), device=actions.device)
                    else:
                        mask = torch.ones((B, B1, 1), device=actions.device)
                    
                    # Collect Q-values for actions from dataset
                    # In T-Maze actions are represented as integers
                    action_indices = actions.long()
                    
                    # Ensure indices are in valid range for gather
                    if action_indices.dim() == 3 and action_indices.shape[-1] == 1:
                        action_indices = action_indices.squeeze(-1)
                    
                    # Check and correct indices
                    max_idx = q1_value.shape[-1] - 1
                    valid_indices = (action_indices >= 0) & (action_indices <= max_idx)
                    
                    # Replace invalid indices with zeros
                    safe_indices = torch.where(valid_indices, action_indices, torch.zeros_like(action_indices))
                    
                    # Expand indices for gather
                    gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, 1)
                    
                    # Collect Q-values
                    q1_selected = q1_value.gather(-1, gather_indices)
                    q2_selected = q2_value.gather(-1, gather_indices)
                    
                    # Compute logsumexp more safely
                    # Mask invalid values
                    q1_for_logsumexp = q1_value.clone()
                    q2_for_logsumexp = q2_value.clone()
                    
                    # For T-Maze CQL loss
                    q1_logsumexp = torch.logsumexp(q1_for_logsumexp, dim=-1, keepdim=True)
                    q2_logsumexp = torch.logsumexp(q2_for_logsumexp, dim=-1, keepdim=True)
                    
                    # Apply mask and compute CQL loss
                    valid_count = mask.sum().item()
                    if valid_count > 0:
                        cql_q1_loss = (q1_logsumexp * mask).sum() / valid_count - (q1_selected * mask).sum() / valid_count
                        cql_q2_loss = (q2_logsumexp * mask).sum() / valid_count - (q2_selected * mask).sum() / valid_count
                        cql_loss = (cql_q1_loss + cql_q2_loss) * self.cql_alpha
                    else:
                        cql_loss = torch.tensor(0.0, device=q1_value.device)
                    
                # For other environments (vizdoom, minigrid_memory, etc.)
                else:
                    # Create a mask of valid values (not padding)
                    if self.padding_idx is not None:
                        if len(actions.shape) == 3 and actions.shape[-1] == 1:
                            mask = (actions.squeeze(-1) != self.padding_idx).float().unsqueeze(-1)
                        else:
                            mask = torch.ones((B, B1, 1), device=actions.device)
                    else:
                        mask = torch.ones((B, B1, 1), device=actions.device)
                    
                    # Get Q-values for actions from dataset
                    if len(actions.shape) == 3 and actions.shape[-1] == 1:
                        # Actions are represented as indices [batch, seq, 1]
                        action_indices = actions.squeeze(-1).long()
                    else:
                        # Actions are represented as one-hot [batch, seq, act_dim]
                        action_indices = actions.argmax(dim=-1)
                    
                    # Check and correct indices
                    max_idx = q1_value.shape[-1] - 1
                    valid_indices = (action_indices >= 0) & (action_indices <= max_idx)
                    safe_indices = torch.where(valid_indices, action_indices, torch.zeros_like(action_indices))
                    
                    # Expand indices for gather
                    gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, 1)
                    
                    # Collect Q-values for selected actions
                    q1_selected = q1_value.gather(-1, gather_indices)
                    q2_selected = q2_value.gather(-1, gather_indices)
                    
                    # Compute logsumexp for all actions
                    q1_logsumexp = torch.logsumexp(q1_value, dim=-1, keepdim=True)
                    q2_logsumexp = torch.logsumexp(q2_value, dim=-1, keepdim=True)
                    
                    # Apply mask and compute CQL loss
                    valid_count = mask.sum().item()
                    if valid_count > 0:
                        cql_q1_loss = (q1_logsumexp * mask).sum() / valid_count - (q1_selected * mask).sum() / valid_count
                        cql_q2_loss = (q2_logsumexp * mask).sum() / valid_count - (q2_selected * mask).sum() / valid_count
                        cql_loss = (cql_q1_loss + cql_q2_loss) * self.cql_alpha
                    else:
                        cql_loss = torch.tensor(0.0, device=q1_value.device)
                    
            except Exception as e:
                print(f"Warning: CQL loss calculation failed: {e}")
                cql_loss = torch.tensor(0.0, device=q1_value.device)
        
        # Inference mode (validation) - simplified logic without CQL loss
        elif actions is None:
            # Simply pass forward without CQL loss
            cql_loss = torch.tensor(0.0, device=q1_value.device) if q1_value.device else torch.tensor(0.0)
        
        # Always use q1_value for actions (BC part)
        logits = q1_value
        
        output = {
            'logits': logits,
            'q1_value': q1_value,
            'q2_value': q2_value,
            'cql_loss': cql_loss,
            'new_mems': None,
            'mem_tokens': None,
            'hidden': new_hidden  # Now return in the same format as BC
        }
        
        return output

    def _get_selected_action_values(self, q_values, actions):
        """
        Safely gets Q-values for selected actions.
        """
        try:
            # For continuous actions (popgym, mikasa_robo, Pendulum)
            if "popgym" in self.env_name or "mikasa_robo" in self.env_name or 'Pendulum' in self.env_name:
                # For continuous actions use MSE between Q and actions
                return q_values.mean(dim=-1, keepdim=True)
                
            # For discrete actions
            elif self.env_name in ['vizdoom', 'minigrid_memory', 'memory_maze', 'tmaze']:
                if len(actions.shape) == 3 and actions.shape[-1] == 1:
                    # Variant 1: actions are represented as indices [batch, seq, 1]
                    action_indices = actions.squeeze(-1).long()
                    # Check that indices are in valid range
                    valid_indices = (action_indices >= 0) & (action_indices < q_values.shape[-1])
                    if not valid_indices.all():
                        # Replace invalid indices with 0
                        action_indices = torch.where(
                            valid_indices, action_indices, 
                            torch.zeros_like(action_indices)
                        )
                    # Collect Q-values for selected actions
                    return q_values.gather(-1, action_indices.unsqueeze(-1))
                
                # Variant 2: actions are represented as one-hot [batch, seq, act_dim]
                elif len(actions.shape) == 3 and actions.shape[-1] > 1:
                    # Get indices of non-zero elements
                    action_indices = actions.argmax(dim=-1)
                    # Check that indices are in valid range
                    valid_indices = (action_indices >= 0) & (action_indices < q_values.shape[-1])
                    if not valid_indices.all():
                        action_indices = torch.where(valid_indices, action_indices, 
                                                   torch.zeros_like(action_indices))
                    # Collect Q-values
                    return q_values.gather(-1, action_indices.unsqueeze(-1))
            
            # Default handling for any other cases
            return q_values.mean(dim=-1, keepdim=True)
            
        except Exception as e:
            print(f"Warning in _get_selected_action_values: {e}")
            # Return average value of Q as a fallback
            return q_values.mean(dim=-1, keepdim=True)

    def reset_hidden(self, batch_size=None, device=None):
        """Reset LSTM hidden states - now in the same format as BC"""
        if self.backbone == 'lstm':
            if batch_size is None or device is None:
                return None
            return self.init_hidden(batch_size, device)
        return None