import torch
import numpy as np

from TMaze_new.TMaze_new_src.utils import set_seed
from collections import defaultdict

import torch
import gymnasium as gym
from ppo_rgb import Agent, Args
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tqdm.notebook import tqdm

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

def get_returns_ManiSkill(model, ret, seed, episode_timeout, context_length, device, act_dim, config, mean, std, use_argmax=False, create_video=False):
    set_seed(seed)

    eval_output_dir = "ManiSkill/rate_val"
    env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos", render_mode="all", sim_backend="gpu")
    eval_envs = gym.make("PushCube-v1", num_envs=1, **env_kwargs)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=True)
    eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=False, 
                                trajectory_name=f"rate_val", max_steps_per_video=100, video_fps=30)
    env = ManiSkillVectorEnv(eval_envs, 1, ignore_terminations=True, record_metrics=True)

    scale = 1
    max_ep_len = episode_timeout#* 3
    
    state_0, _ = env.reset(seed=seed)
    state_0 = state_0['rgb'][0]
    state = torch.tensor(state_0).float().permute(2, 0, 1) # model trained on [C, H, W], but env returns [H, W, C]
    state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
   
    out_states = []
    out_states.append(state.cpu().numpy())
    done = False
    HISTORY_LEN = context_length
    
    rews = []
    frames = [env.render()]
    states = state.to(device=device, dtype=torch.float32)
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
    eval_metrics = defaultdict(list)
    
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
            # print("inp", states_norm.shape, target_return.shape, act_to_pass.shape if act_to_pass is not None else None)
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
            
        if new_mem is not None:
            memories.append(mem_tokens.detach().cpu().numpy())

        act = sampled_action#.detach().cpu().numpy()
        
        actions[-1, :] = act
        act_list.append(act)
        
        state, reward, terminated, truncated, eval_infos = env.step(act) # state [H, W, C], need [C, H, W]
        if "final_info" in eval_infos:
            for k, v in eval_infos["final_info"]["episode"].items():
                eval_metrics[k].append(v)
        state = state['rgb'][0]
        frames.append(env.render())
        state = state.float().permute(2, 0, 1)
        state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
        
        out_states.append(state)
        
        rews.append(reward)
        cur_state = state.to(device)
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

    return episode_return.detach().cpu().numpy().item(), act_list, t, out_states, memories, attn_map, frames, eval_metrics


# ret = 0.8
# SEED = 12



# episode_return, act_list, t, _, _, attn_map, frames = get_returns_ManiSkill(model=model, ret=ret, seed=SEED, episode_timeout=50, 
#                                                                             context_length=config["training_config"]["context_length"], 
#                                                                             device=device, act_dim=config["model_config"]["ACTION_DIM"], 
#                                                                             config=config,
#                                                                             mean=None,
#                                                                             std=None,
#                                                                             use_argmax=config["online_inference_config"]["use_argmax"],
#                                                                             create_video=False)


# print(episode_return)

    # video_path = f"{eval_output_dir}/rate_val.mp4"
    # wandb.log({"episode_video": wandb.Video(video_path)})