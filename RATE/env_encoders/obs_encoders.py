import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    def __init__(self, env_name, state_dim, d_embed):
        super().__init__()

        if env_name == 'tmaze':
            self.obs_encoder = nn.Linear(state_dim, d_embed)
        elif env_name == 'aar':
            self.obs_encoder = nn.Linear(state_dim, d_embed) # * state_dim = 3
        elif env_name == 'memory_maze':
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4, padding=2),
                nn.ReLU(), 
                nn.Conv2d(32, 64, 4, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=2),
                nn.ReLU(),
                nn.Flatten(), 
                nn.Linear(7744, d_embed),
                nn.Tanh(),
            )
        elif env_name == 'minigrid_memory':
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                nn.ReLU(), 
                nn.Flatten(), 
                nn.Linear(3136, d_embed),
            )
        elif env_name == 'vizdoom':
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(), 
                nn.Linear(2560, d_embed),
                nn.Tanh()
            )
        elif env_name == 'atari':
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4, padding=0), 
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0), 
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0), 
                nn.ReLU(),
                nn.Flatten(), 
                nn.Linear(3136, d_embed), 
                nn.Tanh()
            )
        elif env_name == 'mujoco':
            self.obs_encoder = nn.Linear(state_dim, d_embed)
        elif env_name == 'maniskill-pushcube':
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4, padding=0), 
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0), 
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(9216, d_embed),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unknown environment: {env_name}")
