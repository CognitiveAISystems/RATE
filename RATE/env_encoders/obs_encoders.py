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
        elif 'popgym' in env_name:
            self.obs_encoder = nn.Linear(state_dim, d_embed)
        elif "mikasa_robo" in env_name:
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(6, 32, 8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(), 
                nn.Linear(9216, d_embed),
                nn.Tanh()
            )
        elif env_name in ['CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Acrobot-v1', 'Pendulum-v1']:
            # MDP environments with vector observations
            self.obs_encoder = nn.Linear(state_dim, d_embed)
        elif env_name == 'arshot':
            # ARShot environment with token-based observations
            # state_dim is actually the vocab_size for token embeddings
            self.obs_encoder = nn.Embedding(state_dim+1, d_embed)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
