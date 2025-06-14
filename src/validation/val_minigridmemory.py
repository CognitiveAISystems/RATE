import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import numpy as np
import time

from gym import spaces
from gym_minigrid.wrappers import *
from src.utils.set_seed import set_seed

class Minigrid:
    def __init__(self, name, length):
        self._env = gym.make(name)

        self._env.height = length
        self._env.width = length
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        if "Memory" in name:
            view_size = 3
            self.tile_size = 28
            hw = view_size * self.tile_size
            self.max_episode_steps = 96
            self._action_space = spaces.Discrete(3)
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size = self.tile_size)
        else:
            view_size = 7
            self.tile_size = 8
            hw = view_size * self.tile_size
            self.max_episode_steps = 64
            self._action_space = self._env.action_space
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size = self.tile_size)

        self._env = ImgObsWrapper(self._env)

        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3, hw, hw),
                dtype = np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        # This reduces the agent's action space to the only relevant actions (rotate left/right, move forward)
        # to solve the Minigrid-Memory environment.
        return self._action_space
    
    def reset(self, seed=None):
        self._env.seed(np.random.randint(0, 999) if seed is None else seed)
        self.t = 0
        self._rewards = []
        obs = self._env.reset()
        obs = obs.astype(np.float32) / 255.

        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        obs = obs.astype(np.float32) / 255.

        if self.t == self.max_episode_steps - 1:
            done = True

        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        self.t += 1
        return obs, reward, done, info

    def render(self):
        img = self._env.render(tile_size = 96)
        plt.close()
        # time.sleep(0.5)
        return img

    def close(self):
        self._env.close()

def create_env(config:dict, length, render:bool=False):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        env_name {str}: Name of the to be instantiated environment
        render {bool}: Whether to instantiate the environment in render mode. (default: {False})

    Returns:
        {env}: Returns the selected environment instance.
    """

    if config["type"] == "Minigrid":
        return Minigrid(config["name"], length)

@torch.no_grad()
def sample(model, x, block_size, steps, sample=False, top_k=None, actions=None, rtgs=None, timestep=None, mem_tokens=1, saved_context=None, hidden=None):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        
        if saved_context is not None:
            results = model(x_cond, actions, rtgs,None, timestep, *saved_context, mem_tokens=mem_tokens, hidden=hidden)
        else:
            results = model(x_cond, actions, rtgs,None, timestep, mem_tokens=mem_tokens, hidden=hidden) 

        logits = results['logits'][:,-1,:]
        memory = results.get('new_mems', None)
        mem_tokens = results.get('mem_tokens', None)
        hidden = results.get('hidden', None)
        attn_map = getattr(model, 'attn_map', None)
        
    return logits, mem_tokens, memory, attn_map, hidden


def get_returns_MinigridMemory(
        length, model, ret, seed, episode_timeout, context_length, device, 
        config, use_argmax=False, create_video=False,
        env_name={'type': 'Minigrid', 'name': 'MiniGrid-MemoryS13Random-v0'}
    ):
    
    set_seed(seed)
    max_ep_len = episode_timeout

    env = create_env(env_name, length, render=False)
    env.max_episode_steps = episode_timeout
    
    state = torch.tensor(env.reset(seed=seed)).float() * 255.0
    state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, 1), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    
    is_lstm = hasattr(model, 'backbone') and model.backbone in ['lstm', 'gru']

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    saved_context = None
    hidden = model.reset_hidden(1, device) if is_lstm else None

    episode_return, episode_length = 0, 0
    rews = []
    
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, 1), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        if not is_lstm and actions.shape[0] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[slice_index:] if slice_index == 1 else actions[slice_index:,:]
            states = states[:, slice_index:, :]
            target_return = target_return[:,slice_index:]
            timesteps = timesteps[:, slice_index:]
            if t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context

        if is_lstm:
            states = states[:, -1:, :]
            act_to_pass = None if t == 0 else actions[-1:].unsqueeze(0)
            target_return = target_return[:, -1:]#.unsqueeze(-1)
            timesteps = timesteps[:, -1:]
        else:
            states = states
            act_to_pass = None if t == 0 else actions.unsqueeze(0)[:, 1:, :]
            target_return = target_return#.unsqueeze(-1)
            timesteps = timesteps
            if act_to_pass is not None and act_to_pass.shape[1] == 0:
                act_to_pass = None
        
        states_norm = states / 255.0
        
        sample_outputs = sample(
            model=model,  
            x=states_norm,
            block_size=context_length, 
            steps=1, 
            sample=True, 
            actions=act_to_pass, 
            rtgs=target_return.unsqueeze(-1), 
            timestep=timesteps, 
            mem_tokens=mem_tokens,
            saved_context=saved_context,
            hidden=hidden
        )

        sampled_action, new_mem_tokens, new_context, attn_map, new_hidden = sample_outputs

        if is_lstm:
            hidden = new_hidden
            
        action_probs = torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy()
        if not use_argmax:
            act = np.random.choice([0, 1, 2, 3], p=action_probs)
        else:
            act = np.argmax(action_probs)

        actions[-1] = act
        
        state, reward, done, info = env.step((act, ))
        state = np.float32(state) * 255.
        state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
        rews.append(reward)
        cur_state = torch.from_numpy(state).to(device=device).float()
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = reward
        pred_return = target_return[0,-1] - reward
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (1)], dim=1)
        episode_return += reward
        episode_length += 1
        
        if done:
            break  
    
    env.close()

    return episode_return, t