import os
import sys
sys.path.append("../../")
sys.path.append("../../../")
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from RATE import mem_transformer_v2_GTrXL
# from RATE_GTrXL import mem_transformer_v2_GTrXL_v2_ca2
import os
import torch
import numpy as np
from tqdm import tqdm

from VizDoom.VizDoom_src.utils import env_vizdoom2

# import matplotlib.pyplot as plt
import random
# import seaborn as sns

import matplotlib.pyplot as plt
# from itertools import count
# import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

import yaml
with open(f"MemoryMaze/MemoryMaze_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config["model_mode"] = 'RATE'
config["arctitecture_mode"] = 'TrXL'
# config["training_config"]["sections"] = 3

""" ARCHITECTURE MODE """
if config["arctitecture_mode"] == "TrXL":
    config["model_config"]["use_gate"] = False
    config["model_config"]["use_stable_version"] = False
elif config["arctitecture_mode"] == "TrXL-I":
    config["model_config"]["use_gate"] = False
    config["model_config"]["use_stable_version"] = True
elif config["arctitecture_mode"] == "GTrXL":
    config["model_config"]["use_gate"] = True
    config["model_config"]["use_stable_version"] = True     

print(f"Selected Architecture: {config['arctitecture_mode']}")  

""" MODEL MODE """
if config["model_mode"] == "RATE": 
    config["model_config"]["mem_len"] = 360 ########################### 2 FOR DTXL 0
    config["model_config"]["mem_at_end"] = True ########################### True FOR DTXL False
# elif config["model_mode"] == "DT":
#     config["model_config"]["mem_len"] = 0 ########################### 2 FOR DTXL 0
#     config["model_config"]["mem_at_end"] = False ########################### True FOR DTXL False
#     config["model_config"]["num_mem_tokens"] = 0
# elif config["model_mode"] == "DTXL":
#     config["model_config"]["mem_len"] = 2
#     config["model_config"]["mem_at_end"] = False
#     config["model_config"]["num_mem_tokens"] = 0
# elif config["model_mode"] == "RATEM":
#     print("Custom Mode!!! RATE wo cache (RMT)")
#     config["model_config"]["mem_len"] = 0
#     config["model_config"]["mem_at_end"] = True
# elif config["model_mode"] == "RATE_wo_nmt":
#     print("Custom Mode!!! RATE wo nmt")
#     config["model_config"]["mem_len"] = 2
#     config["model_config"]["mem_at_end"] = False
#     config["model_config"]["num_mem_tokens"] = 0
#     max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # it's ruine np.random.choice()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
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
            
        logits = results[0][0].detach()[:,-1,:]
        mem_tokens = results[1]
        memory = results[0][2:]
        attn_map = model.attn_map
        
    return logits, mem_tokens, memory, attn_map
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import logging
logging.disable(logging.WARNING)
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
import warnings
warnings.filterwarnings('ignore')

from TMaze_new.TMaze_new_src.utils import set_seed
from VizDoom.VizDoom_src.utils import z_normalize, inverse_z_normalize



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
            
        logits = results[0][0].detach()[:,-1,:]
        mem_tokens = results[1]
        memory = results[0][2:]
        attn_map = model.attn_map
        
    return logits, mem_tokens, memory, attn_map

def get_returns_MemoryMaze(model, ret, seed, episode_timeout, context_length, device, act_dim, config, mean, std, use_argmax=False, create_video=False):
    
    set_seed(seed)
    
    # * USE ONLY LAST MEM TOKEN:
    use_only_last_mem_token = False
    
    # * ADD NOISE TO MEM TOKENS:
    add_noise = False

    scale = 1
    max_ep_len = episode_timeout#* 3

    env = gym.make('memory_maze:MemoryMaze-9x9-v0', seed=seed)
    
    state_0 = env.reset()
    state = torch.tensor(state_0).float().permute(2, 0, 1) # model trained on [C, H, W], but env returns [H, W, C]
    state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
   
    out_states = []
    out_states.append(state.cpu().numpy())
    done = True
    HISTORY_LEN = context_length
    
    rews = []
    frames = [state_0]
    states = state.to(device=device, dtype=torch.float32)
    act_dim = 1
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    episode_return, episode_length = 0, 0

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    saved_context = None
    segment = 0
    prompt_steps = 0
    
    act_list= []
    memories = []
    
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
                    timesteps = timesteps[:, -1:]
                else:
                    actions = actions[-prompt_steps:,:]
                    states = states[:, -prompt_steps:, :, :, :]
                    target_return = target_return[:,-prompt_steps:]
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
                    target_return = target_return[:,-prompt_steps:]
                    timesteps = timesteps[:, -prompt_steps:]
                    
                if t%(context_length)==0 and t>5:
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                    mem_tokens = new_mem
                    saved_context = new_notes

        if t==0:
            act_to_pass = None
        else:
            act_to_pass = actions.unsqueeze(0)[:, 1:, :]
            if act_to_pass.shape[1] == 0:
                act_to_pass = None 
        
        if add_noise:
            added_noise = torch.randn_like(mem_tokens)
        else:
            added_noise = torch.zeros_like(mem_tokens) if mem_tokens is not None else 0
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!
        if config["data_config"]["normalize"] == 2:
            b, l, c, h, w = states.shape
            states_norm = states.reshape(b*l, c, h, w)
            states_norm = z_normalize(states_norm, mean.to(device), std.to(device))
            states_norm = states_norm.reshape(b, l, c, h, w).to(device)
        elif config["data_config"]["normalize"] == 1:
            states_norm = states / 255.0
        elif config["data_config"]["normalize"] == 0:
            states_norm = states

        # !!!!!!!!!!!!!!!!!!!
        try:
            sampled_action, new_mem, new_notes, attn_map = sample(model=model,  
                                                        x=states_norm,
                                                        block_size=HISTORY_LEN, 
                                                        steps=1, 
                                                        sample=True, 
                                                        actions=act_to_pass, 
                                                        rtgs=target_return.unsqueeze(-1), 
                                                        timestep=timesteps, 
                                                        mem_tokens=mem_tokens,
                                                        saved_context=saved_context)
        except:
            print("ERROR!!!!!!!!!!!", t, states.shape, target_return.shape, act_to_pass.shape)
            
        # !!!!!!!
        if use_only_last_mem_token:
            if t > 0 and t % (context_length-1) == 0 and switcher == False:
                switcher = True
                saved_mem = new_mem
            
        if new_mem is not None:
            memories.append(mem_tokens.detach().cpu().numpy())
            
        if not use_argmax:
            act = np.random.choice(np.arange(env.action_space.n), p=torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        else:
            act = np.argmax(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())

        # print(act)
        
        actions[-1] = act
        act_list.append(act)
        
        state, reward, done, info = env.step(act) # state [H, W, C], need [C, H, W]
        frames.append(state)
        state = np.float32(state.transpose(2, 0, 1))
        state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
        
        out_states.append(state)
        
        rews.append(reward)
        #print(reward)
        cur_state = torch.from_numpy(state).to(device=device).float()
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = reward
        pred_return = target_return[0,-1] - (reward/scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (1)], dim=1)
        episode_return += reward
        episode_length += 1
        
        if done:
            torch.cuda.empty_cache()
            break  
        
    if create_video == True:
        print("\n")
    
    env.close()
    return episode_return, act_list, t, out_states, memories, attn_map, frames

# * ##############################################

DESIRED_RETURN = 18.1
TOTAL_RETURNS = []


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

ckpt_path = 'MemoryMaze/MemoryMaze_checkpoints/wo_filter/v2_RATE/wo_filter_RATE_arch_mode_TrXL_RATE_RUN_1_2024_11_30_16_31_52/137250000_KTD.pth'
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device)
_ = model.eval()

for ret in [DESIRED_RETURN]:
    print(ckpt_path)
    print("RET:", ret)

    returns = []

    pbar = tqdm(range(100))
    ts = []
    mean_ret = 0
    sum_ret = 0
    cnt = 1
    for seed in pbar:
        episode_return, act_list, t, out_states, memories, attn_map, frames = get_returns_MemoryMaze(model=model, ret=ret, seed=seed, episode_timeout=1000, context_length=config["training_config"]["context_length"], 
                                                            device=device, act_dim=config["model_config"]["ACTION_DIM"], config=config, mean=0, std=0, use_argmax=False, create_video=False)
        returns.append(episode_return)
        sum_ret += episode_return
        mean_ret = sum_ret / cnt

        pbar.set_description(f"Time: {t}, Return: {episode_return:.2f}, Mean Return {mean_ret:.2f}")
        cnt += 1

mean_return = np.mean(returns)
TOTAL_RETURNS.append(mean_return)


print('Mean Return:', mean_return)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

# ckpt_path = f'{parent_dir}/MemoryMaze/MemoryMaze_checkpoints/RATE/RATE_arch_mode_TrXL_RATE_RUN_2_2024_09_16_04_48_08/44178750_KTD.pth'
# model.load_state_dict(torch.load(ckpt_path, map_location=device))
# model.to(device)
# _ = model.eval()

# for ret in [DESIRED_RETURN]:
#     print(ckpt_path)
#     print("RET:", ret)

#     returns = []

#     pbar = tqdm(range(100))
#     ts = []
#     mean_ret = 0
#     sum_ret = 0
#     cnt = 1
#     for seed in pbar:
#         episode_return, act_list, t, out_states, memories, attn_map, frames = get_returns_MemoryMaze(model=model, ret=ret, seed=seed, episode_timeout=1000, context_length=config["training_config"]["context_length"], 
#                                                             device=device, act_dim=config["model_config"]["ACTION_DIM"], config=config, mean=0, std=0, use_argmax=False, create_video=False)
#         returns.append(episode_return)
#         sum_ret += episode_return
#         mean_ret = sum_ret / cnt

#         pbar.set_description(f"Time: {t}, Return: {episode_return:.2f}, Mean Return {mean_ret:.2f}")
#         cnt += 1

# mean_return = np.mean(returns)
# TOTAL_RETURNS.append(mean_return)


# print('Mean Return:', mean_return)

# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

# ckpt_path = f'{parent_dir}/MemoryMaze/MemoryMaze_checkpoints/RATE/RATE_arch_mode_TrXL_RATE_RUN_3_2024_09_16_08_37_57/40162500_KTD.pth'
# model.load_state_dict(torch.load(ckpt_path, map_location=device))
# model.to(device)
# _ = model.eval()

# for ret in [DESIRED_RETURN]:
#     print(ckpt_path)
#     print("RET:", ret)

#     returns = []

#     pbar = tqdm(range(100))
#     ts = []
#     mean_ret = 0
#     sum_ret = 0
#     cnt = 1
#     for seed in pbar:
#         episode_return, act_list, t, out_states, memories, attn_map, frames = get_returns_MemoryMaze(model=model, ret=ret, seed=seed, episode_timeout=1000, context_length=config["training_config"]["context_length"], 
#                                                             device=device, act_dim=config["model_config"]["ACTION_DIM"], config=config, mean=0, std=0, use_argmax=False, create_video=False)
#         returns.append(episode_return)
#         sum_ret += episode_return
#         mean_ret = sum_ret / cnt

#         pbar.set_description(f"Time: {t}, Return: {episode_return:.2f}, Mean Return {mean_ret:.2f}")
#         cnt += 1

# mean_return = np.mean(returns)
# TOTAL_RETURNS.append(mean_return)


# print('Mean Return:', mean_return)

# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
from scipy.stats import sem

print('*'*100)

mean_ = np.mean(TOTAL_RETURNS)
sem_ = sem(TOTAL_RETURNS)
print(f'Final: {mean_:.2f} ± {sem_:.2f}')