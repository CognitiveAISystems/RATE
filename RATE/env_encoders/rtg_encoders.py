import torch
import torch.nn as nn


class RTGEncoder(nn.Module):
    def __init__(self, env_name, d_embed):
        super().__init__()
        self.env_name = env_name
        self.d_embed = d_embed

        if env_name == 'tmaze':
            self.rtg_encoder = nn.Linear(1, d_embed)
        elif env_name == 'aar':
            self.rtg_encoder = nn.Linear(1, d_embed)
        elif env_name == 'memory_maze':
            self.rtg_encoder = nn.Sequential(nn.Linear(1, d_embed), nn.Tanh()) 
        elif env_name == 'minigrid_memory':
            self.rtg_encoder = nn.Sequential(nn.Linear(1, d_embed))
        elif env_name == 'vizdoom':
            self.rtg_encoder = nn.Sequential(nn.Linear(1, d_embed), nn.Tanh())
        elif env_name == 'atari':
            self.rtg_encoder = nn.Sequential(nn.Linear(1, d_embed), nn.Tanh())
        elif env_name == 'mujoco':
            self.rtg_encoder = nn.Linear(1, d_embed)
        elif 'popgym' in env_name:
            self.rtg_encoder = nn.Linear(1, d_embed)
        elif "mikasa_robo" in env_name:
            self.rtg_encoder = nn.Linear(1, d_embed)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
