import datetime
import wandb
from torch.utils.data import random_split, DataLoader
import argparse
import yaml
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import sys


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
        dropout=0.0,
        num_layers=2,
        mode='mujoco',
        arch_mode='lstm',
        device='cuda'
    ):
        super().__init__(state_dim, act_dim, max_length=max_length) 
       
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_layers = num_layers
        self.arch_mode = arch_mode

        self.device = device


        self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )

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
            self.embed_action_toq = nn.Sequential(nn.Embedding(self.act_dim, self.act_dim), nn.Tanh())
            self.embed_action = nn.Sequential(nn.Embedding(self.act_dim, hidden_size), nn.Tanh())
            self.embed_return = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh())
        
        if self.mode == 'memory_maze':
            self.state_dim = 3
            self.act_dim = 6
            self.predict_action = nn.Linear(hidden_size, self.act_dim, bias=False)
            self.embed_state = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4, padding=2),
                                    nn.ReLU(), 
                                    nn.Conv2d(32, 64, 4, stride=2, padding=2),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.Flatten(), nn.Linear(7744, hidden_size), 
                                    nn.Tanh())

            self.embed_action = nn.Sequential(nn.Embedding(self.act_dim, hidden_size), nn.Tanh())
            self.embed_return = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh()) 

        if self.mode == 'minigrid_memory':
            self.state_dim = 3
            self.act_dim = 4
            self.predict_action = nn.Linear(hidden_size, self.act_dim, bias=False)
            self.embed_state = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4, padding=0),
                                           	nn.ReLU(),
                                           	nn.Conv2d(32, 64, 4, stride=2, padding=0),
                                           	nn.ReLU(),
                                           	nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                           	nn.ReLU(), 
                                           	nn.Flatten(), nn.Linear(3136, hidden_size))
            self.embed_action = nn.Sequential(nn.Embedding(self.act_dim, hidden_size))
            self.embed_return = nn.Sequential(nn.Linear(1, hidden_size))

        self.q1 = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.cql_alpha = 1.0


    def init_hidden(self, batch_size): 
        self.h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        self.c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 


    def forward(self, states, actions, returns_to_go, attention_mask=None, update_hidden=True, stacked_input=False):
        
        
        if self.mode == 'tmaze':
            batch_size, seq_length = states.shape[0], states.shape[1]
            state_embeddings = self.embed_state(states)

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

            returns_embeddings = self.embed_return(returns_to_go)
            

        if self.mode in ['doom', 'memory_maze', 'minigrid_memory']:
            batch_size, seq_length, C, H, W = states.shape
            states = states.reshape(-1, C, H, W).type(torch.float32).contiguous() 
            state_embeddings = self.embed_state(states).reshape(batch_size, seq_length, self.hidden_size)

            if self.mode == 'minigrid_memory':
                actions = torch.where(actions == -10, torch.tensor(3), actions)
                actions = actions.to(dtype=torch.long, device=states.device)
            action_embeddings = self.embed_action(actions).squeeze(-2)
            returns_embeddings = self.embed_return(returns_to_go)

        if stacked_input:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, returns_embeddings), dim=1
            ).reshape(batch_size, 3*seq_length, self.hidden_size)
        else:
            stacked_inputs = state_embeddings


        if update_hidden:
            lstm_outputs, (self.h_0, self.c_0) = self.lstm(stacked_inputs, (self.h_0, self.c_0))
        else:
            lstm_outputs, (_, _) = self.lstm(stacked_inputs, (self.h_0, self.c_0))

        if stacked_input:
            lstm_outputs = lstm_outputs[:,0::3,:]
        else:
            lstm_outputs = lstm_outputs

        action_preds = self.predict_action(lstm_outputs)

        state_action = torch.cat([lstm_outputs, actions], dim=-1)
        q1 = self.q1(state_action)
        q2 = self.q2(state_action)

        # CQL regularization
        random_actions = torch.randint(0, self.act_dim, (batch_size, seq_length, 1)).to(self.device)
        random_state_action = torch.cat([lstm_outputs, random_actions], dim=-1)
        random_q1 = self.q1(random_state_action)
        random_q2 = self.q2(random_state_action)

        cql_loss = (torch.logsumexp(random_q1, dim=1) - q1.mean(dim=1)).mean() + \
                   (torch.logsumexp(random_q2, dim=1) - q2.mean(dim=1)).mean()

        return action_preds, q1, q2, self.cql_alpha * cql_loss

