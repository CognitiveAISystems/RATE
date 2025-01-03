import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import wandb

class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
    

class MultiplyByScalarLayer(nn.Module):
    # A simple layer to multiply all entries by a constant scalar value. Needed since action inputs are not normalized in
    # many environments and tanh is then critical, unlike in D4RL where actions are in [-1, 1].
    # scalar value should be absolute max possible action value.

    def __init__(self, scalar):
        super(MultiplyByScalarLayer, self).__init__()
        self.scalar = scalar

    def forward(self, tensors):
        result = torch.clone(tensors)
        for ind in range(result.shape[0]):
            result[ind] = torch.mul(result[ind], self.scalar)
        return result
    
class DecisionLSTM(TrajectoryModel):

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        dropout=0.1,
        num_layers=1,
        mode='mujoco',
        arch_mode='lstm'
    ):
        super().__init__(state_dim, act_dim, max_length=max_length) 
       
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_layers = num_layers
        self.arch_mode = arch_mode

        if self.arch_mode == 'lstm':
            # LSTM
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        elif self.arch_mode == 'gru':
            # LSTM
            self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise NotImplementedError('Incorrect architecture mode input!')

        self.embed_ln = nn.LayerNorm(hidden_size)

        if self.mode == 'tmaze':
            self.state_dim = 4
            self.act_dim = 1
            self.predict_action = nn.Linear(hidden_size, 4, bias=False)
            self.embed_action = nn.Sequential(nn.Embedding(4+1, hidden_size), nn.Tanh())
            self.embed_return = nn.Linear(1, hidden_size)
            self.embed_state = nn.Linear(4, hidden_size)

        if self.mode == 'doom':
            self.state_dim = 3
            self.act_dim = 5
            self.predict_action = nn.Linear(hidden_size, self.act_dim, bias=False)
            self.embed_state = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4, padding=0),
                                            nn.ReLU(),
                                            nn.Conv2d(32, 64, 4, stride=2, padding=0),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                            nn.ReLU(),
                                            nn.Flatten(), nn.Linear(2560, hidden_size),
                                            nn.Tanh())
            self.embed_action = nn.Sequential(nn.Embedding(self.act_dim, hidden_size), nn.Tanh())
            self.embed_return = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh())


        self.component_names = {
            'embed_state': self.embed_state,
            'embed_action': self.embed_action,
            'embed_return': self.embed_return,
            'embed_ln': self.embed_ln,
            'predict_action': self.predict_action
        }
        if self.arch_mode == 'lstm':
            self.component_names['lstm'] = self.lstm
        else:
            self.component_names['gru'] = self.gru

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, wwandb=False):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if self.mode == 'tmaze':
            # embed each modality with a different head
            state_embeddings = self.embed_state(states)
            # action_embeddings = self.embed_action(actions)
            returns_embeddings = self.embed_return(returns_to_go)  

            if actions is not None:
                use_long = False
                for name, module in self.embed_action.named_children():
                    if isinstance(module, nn.Embedding):
                        use_long = True
                if use_long:
                    if self.mode == 'tmaze':
                        actions = torch.where(actions == -10, torch.tensor(4), actions)
                    elif self.mode == 'aar':
                        actions = torch.where(actions == -10, torch.tensor(3), actions)
                    actions = actions.to(dtype=torch.long, device=states.device)
                    action_embeddings = self.embed_action(actions).squeeze(2) # (batch, block_size, n_embd)
                else:
                    action_embeddings = self.embed_action(actions) # (batch, block_size, n_embd)
        else:
            if self.mode == 'doom':
                B, B1, C, H, W = states.shape
                states = states.reshape(-1, C, H, W).type(torch.float32).contiguous() 
            else:
                raise NotImplementedError('Incorrect mode input!')

            # state_embeddings = self.embed_state(states).reshape(B, B1, self.hidden_size)
            state_embeddings = self.embed_state(states).reshape(B1, B, self.hidden_size)
            returns_embeddings = self.embed_return(returns_to_go).reshape(B1, B, self.hidden_size)

            if actions is not None:
                use_long = False
                for name, module in self.embed_action.named_children():
                    if isinstance(module, nn.Embedding):
                        use_long = True
                if use_long:
                    if self.mode != 'minigrid_memory':
                        actions = actions.to(dtype=torch.long, device=states.device)
                        action_embeddings = self.embed_action(actions).squeeze(2) # (batch, block_size, n_embd)
                    action_embeddings = self.embed_action(actions).squeeze(2) # (batch, block_size, n_embd)
                else:
                    action_embeddings = self.embed_action(actions) # (batch, block_size, n_embd)
            action_embeddings = action_embeddings.reshape(B1, B, self.hidden_size)

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(1, 0, 2, 3).reshape(3*seq_length, batch_size, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)     

        h_0 = Variable(torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        )).to(states.device)
        if self.arch_mode == 'lstm':
            c_0 = Variable(torch.zeros(
                self.num_layers, batch_size, self.hidden_size
            )).to(states.device)

        if self.arch_mode == 'lstm':
            lstm_outputs, _ = self.lstm(
                stacked_inputs,
                (h_0, c_0)
            )
        elif self.arch_mode == 'gru':
            lstm_outputs, _ = self.gru(
                stacked_inputs,
                h_0
            )
        else:
            raise NotImplementedError('Incorrect architecture mode input!')
        
        x = lstm_outputs
        x = lstm_outputs.reshape(3, seq_length, batch_size, self.hidden_size)
        x = x.permute(2, 0, 1, 3)  # [batch, 3, seq, hidden]

        action_preds = self.predict_action(x[:,1])  # predict next action given state

        if self.training and wwandb:
            self.log_gradient_norms(wwandb=wwandb)

        return action_preds

    def log_gradient_norms(self, wwandb=True):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if wwandb:
            wandb.log({"gradient_norm/total": total_norm})

        for name, component in self.component_names.items():
            component_norm = 0
            for p in component.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    component_norm += param_norm.item() ** 2
            component_norm = component_norm ** 0.5
            
            if wwandb:
                wandb.log({f"gradient_norm/{name}": component_norm})

        return total_norm

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        if self.mode == 'doom':
            B, B1, C, H, W = states.shape
        else:
            states = states.reshape(1, -1, self.state_dim)
        if self.mode != 'doom':
            actions = actions.reshape(1, -1, self.act_dim)
        else:
            actions = actions.reshape(1, -1, 1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            attention_mask = None
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            if self.mode == 'doom':
                states = torch.cat(
                    [torch.zeros((states.shape[0], self.max_length-states.shape[1], C, H, W), device=states.device), states],
                    dim=1).to(dtype=torch.float32)
            else:
                states = torch.cat(
                    [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                    dim=1).to(dtype=torch.float32)
            
            if self.mode != 'doom':
                actions = torch.cat(
                    [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                                device=actions.device), actions],
                    dim=1).to(dtype=torch.float32)
            else:
                actions = torch.cat(
                    [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], 1),
                                device=actions.device), actions],
                    dim=1).to(dtype=torch.float32)
                
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]