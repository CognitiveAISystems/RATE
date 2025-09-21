import torch
import torch.nn as nn


class ActDecoder(nn.Module):
    def __init__(self, env_name, act_dim, d_embed):
        super().__init__()
        self.env_name = env_name
        self.act_dim = act_dim
        self.d_embed = d_embed

        if env_name == 'tmaze':
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False) # * act_dim = 4
        elif env_name == 'aar':
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False) # * act_dim = 3
        elif env_name == 'memory_maze':
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False) # * act_dim = 6
        elif env_name == 'minigrid_memory':
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False) # * act_dim = 4
        elif env_name == 'vizdoom':
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False) # * act_dim = 5
        elif env_name == 'atari':
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False) # * act_dim depends on the env
        elif env_name == 'mujoco':
            self.act_decoder = nn.Sequential(
                nn.Linear(d_embed, act_dim), 
                nn.Tanh(),
            )
        elif 'popgym' in env_name:
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False) # * act_dim depends on the env
        elif "mikasa_robo" in env_name:
            self.act_decoder = nn.Sequential(
                nn.Linear(d_embed, act_dim), 
                nn.Tanh(),
            )
        elif env_name in ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']:
            # Discrete MDP environments
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False)
        elif env_name in ['MountainCarContinuous-v0', 'Pendulum-v1']:
            # Continuous MDP environments
            self.act_decoder = nn.Sequential(
                nn.Linear(d_embed, act_dim), 
                nn.Tanh(),
            )
        elif env_name == 'arshot':
            # ARShot environment with token-based actions
            # act_dim is actually the vocab_size for token prediction
            self.act_decoder = nn.Linear(d_embed, act_dim, bias=False)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
