import torch
import torch.nn as nn



class ActEncoder(nn.Module):
    def __init__(self, env_name, act_dim, d_embed):
        super().__init__()
        self.env_name = env_name
        self.act_dim = act_dim
        self.d_embed = d_embed

        if env_name == 'tmaze':
            self.act_encoder = nn.Sequential(nn.Embedding(act_dim+1, d_embed), nn.Tanh()) # * act_dim = 4
        elif env_name == 'aar':
            self.act_encoder = nn.Sequential(nn.Embedding(act_dim+1, d_embed), nn.Tanh()) # * act_dim = 3
        elif env_name == 'memory_maze':
            self.act_encoder = nn.Sequential(nn.Embedding(act_dim, d_embed), nn.Tanh()) # * act_dim = 6
        elif env_name == 'minigrid_memory':
            self.act_encoder = nn.Sequential(nn.Embedding(act_dim+1, d_embed)) # * act_dim = 4
        elif env_name == 'vizdoom':
            self.act_encoder = nn.Sequential(nn.Embedding(act_dim, d_embed), nn.Tanh()) # * act_dim = 5
        elif env_name == 'atari':
            self.act_encoder = nn.Sequential(nn.Embedding(act_dim, d_embed), nn.Tanh()) # * act_dim depends on the env
        elif env_name == 'mujoco':
            self.act_encoder = nn.Linear(act_dim, d_embed)
        elif 'popgym' in env_name:
            if any(char in env_name for char in ['NoisyPositionOnlyPendulumMedium']):
                self.act_encoder = nn.Linear(act_dim, d_embed)
            else:
                self.act_encoder = nn.Sequential(nn.Embedding(act_dim+1, d_embed)) # * act_dim depends on the env
        elif "mikasa_robo" in env_name:
            self.act_encoder = nn.Linear(act_dim, d_embed) # * act_dim = 8
        else:
            raise ValueError(f"Unknown environment: {env_name}")