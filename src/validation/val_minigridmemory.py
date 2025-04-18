import torch
import numpy as np

from src.utils.set_seed import set_seed
import matplotlib.pyplot as plt

import gym
import numpy as np
import time

from gym import spaces
from gym_minigrid.wrappers import *

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
def sample(model, x, block_size, steps, sample=False, top_k=None, actions=None, rtgs=None, timestep=None, mem_tokens=1, saved_context=None):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        if saved_context is not None:
            results = model(x_cond, actions, rtgs,None, timestep, *saved_context, mem_tokens=mem_tokens)
        else:
            results = model(x_cond, actions, rtgs,None, timestep, mem_tokens=mem_tokens) 

        logits = results['logits'][:,-1,:]
        memory = results['new_mems']
        mem_tokens = results['mem_tokens']
        
        attn_map = model.attn_map
        
    return logits, mem_tokens, memory, attn_map


def get_returns_MinigridMemory(length, model, ret, seed, episode_timeout, context_length, device, 
                               config, use_argmax=False, create_video=False,
                               env_name={'type': 'Minigrid', 'name': 'MiniGrid-MemoryS9-v0'}):
    
    set_seed(seed)
    
    # * USE ONLY LAST MEM TOKEN:
    use_only_last_mem_token = False
    
    # * ADD NOISE TO MEM TOKENS:
    add_noise = False

    scale = 1
    max_ep_len = episode_timeout#* 3

        
    # env_name = {'type': 'Minigrid', 'name': "MiniGrid-MemoryS17Random-v0"}
    # env_name = {'type': 'Minigrid', 'name': 'MiniGrid-MemoryS9-v0'} # train data
    env = create_env(env_name, length, render=True)
    
    state = torch.tensor(env.reset(seed=seed)).float() * 255.
    plt.close()
    state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
   
    out_states = []
    out_states.append(state.cpu().numpy())
    done = True
    frames = []
    HISTORY_LEN = context_length 
    
    rews = []
    act_dim = 1
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    episode_return, episode_length = 0, 0

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    saved_context = None
    segment = 0
    prompt_steps = 0# 5
    
    act_list= []
    memories = []

    attn_maps = []
    attn_maps_seg = []

    frames = []
    
    # !!!
    if use_only_last_mem_token:
        switcher = False
        saved_mem = None
    
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        if config["model_mode"] != 'DT' and config["model_mode"] != 'DTXL':
            if actions.shape[0] > HISTORY_LEN:
                segment+=1
                
                if prompt_steps==0:
                    actions = actions[-1:,:]
                    states = states[:, -1:, :, :, :]
                    target_return = target_return[:,-1:]
                    #target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
                    timesteps = timesteps[:, -1:]
                else:
                    actions = actions[-prompt_steps:,:]
                    states = states[:, -prompt_steps:, :, :, :]
                    target_return = target_return[:,-prompt_steps:]#+3600.
                    timesteps = timesteps[:, -prompt_steps:]
                    
                if t%(context_length)==0 and t>5:
                    # !!!
                    if use_only_last_mem_token:
                        mem_tokens = saved_mem
                    else:
                        mem_tokens = new_mem
                    
                    saved_context = new_notes
                    
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                    
        else:
            if actions.shape[0] > HISTORY_LEN:
                segment+=1
                
                if prompt_steps==0:
                    actions = actions[1:,:]
                    states = states[:, 1:, :, :, :]
                    target_return = target_return[:,1:]
                    timesteps = timesteps[:, 1:]
                else:
                    actions = actions[-prompt_steps:,:]
                    states = states[:, -prompt_steps:, :, :, :]
                    target_return = target_return[:,-prompt_steps:]#+3600.
                    timesteps = timesteps[:, -prompt_steps:]
                    
                if t%(context_length)==0 and t>5: 
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                    mem_tokens = new_mem
                    saved_context = new_notes

        #print(states.shape,actions.shape,target_return.shape,timesteps.shape)
        if t==0:
            act_to_pass = None
        else:
            act_to_pass = actions.unsqueeze(0)[:, 1:, :]
            if act_to_pass.shape[1] == 0:
                act_to_pass = None 
        
        #print(states.shape, target_return.shape, act_to_pass.shape if act_to_pass is not None else act_to_pass)
        # print(t, timesteps[:, -1])
        
        if add_noise:
            added_noise = torch.randn_like(mem_tokens)
        else:
            added_noise = torch.zeros_like(mem_tokens) if mem_tokens is not None else 0
        
        states_norm = states / 255.0
        
        try:
            sampled_action, new_mem, new_notes, attn_map = sample(model=model,  
                                                        x=states_norm,
                                                        block_size=HISTORY_LEN, 
                                                        steps=1, 
                                                        sample=True, 
                                                        actions=act_to_pass, 
                                                        rtgs=target_return.unsqueeze(-1), 
                                                        timestep=timesteps, 
                                                        mem_tokens=mem_tokens,#torch.randn_like(mem_tokens),#mem_tokens+added_noise, 
                                                        saved_context=saved_context)
        except:
            pass
            #print("ERROR!!!!!!!!!!!", t, states.shape, target_return.shape, act_to_pass.shape)
            
        # !!!!!!!
        if use_only_last_mem_token:
            if t > 0 and t % (context_length-1) == 0 and switcher == False:
                switcher = True
                saved_mem = new_mem
            
        if new_mem is not None:
            memories.append(mem_tokens.detach().cpu().numpy())
        #print(sampled_action[-1])
        
        
        # print(np.round(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy(), 3))
        # * if OHE
        # if not use_argmax:
        #     act = np.random.choice([3, 4], p=torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        # else:
        #     act = np.argmax(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        #     if act == 0:
        #         act = 3
        #     elif act == 1:
        #         act = 4
        # *else:
        if not use_argmax:
            act = np.random.choice([0, 1, 2, 3], p=torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        else:
            act = np.argmax(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())        

        #print(act, actions.shape, torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        actions[-1] = act
        act_list.append(act)

        # Render environment
        if create_video:
            rofl = env.render()
            plt.close()
            frames.append(rofl)
        
        state, reward, done, info = env.step((act, ))
        #print(t, reward)
        state = np.float32(state) * 255.
        state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
        
        out_states.append(state)
        
        rews.append(reward)
        #print(reward)
        cur_state = torch.from_numpy(state).to(device=device).float()
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = reward
        pred_return = target_return[0,-1] - (reward/scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        #target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
        # timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (1)], dim=1)
        episode_return += reward
        episode_length += 1

        attn_maps.append(attn_map)
    
        if (t+1) % (context_length) == 0 and t > 0:
            attn_maps_seg.append(attn_map)
        
        if done:
            torch.cuda.empty_cache()
            break  
        
    if create_video == True:
        print("\n")

    # after done, render last state
    if create_video:
        env.render()
        plt.close()
        frames.append(rofl)
    
    env.close()

    return episode_return, act_list, t, out_states, memories, attn_maps, attn_maps_seg, frames
