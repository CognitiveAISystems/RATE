import torch
import numpy as np

from TMaze_new.TMaze_new_src.utils import set_seed
from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Any

import torch
import gymnasium as gym
from ppo_rgb import Agent, Args
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tqdm.notebook import tqdm

import threading

# import logging
# logging.getLogger('mani_skill').setLevel(logging.ERROR)

# import warnings
# warnings.filterwarnings('ignore', message='No initial pose set for actor builder of goal_region.*')
# warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.core')

# logging.getLogger('mani_skill').disabled = True

@torch.no_grad()
def sample(
    model: torch.nn.Module,
    x: torch.Tensor,
    block_size: int,
    steps: int = 1,
    sample: bool = False,
    actions: Optional[torch.Tensor] = None,
    rtgs: Optional[torch.Tensor] = None,
    timestep: Optional[torch.Tensor] = None,
    mem_tokens: Optional[torch.Tensor] = None,
    saved_context: Optional[List] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List], Any]:
    """Sample actions from the model.
    
    Args:
        model: The transformer model
        x: Input states tensor
        block_size: Maximum sequence length
        steps: Number of sampling steps
        sample: Whether to sample or take argmax
        actions: Previous actions tensor
        rtgs: Returns-to-go tensor  
        timestep: Current timestep tensor
        mem_tokens: Memory tokens
        saved_context: Saved context from previous forward pass
    
    Returns:
        Tuple of (logits, new_mem_tokens, new_context, attention_map)
    """
    model.eval()
    
    # Crop sequences if needed
    x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
    if actions is not None:
        actions = actions if actions.size(1) <= block_size else actions[:, -block_size:]
    rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:]

    # Forward pass
    if saved_context is not None:
        results = model(x_cond, actions, rtgs, None, timestep, *saved_context, mem_tokens=mem_tokens)
    else:
        results = model(x_cond, actions, rtgs, None, timestep, mem_tokens=mem_tokens)

    return (
        results[0][0].detach()[:,-1,:],  # logits
        results[1],                       # mem_tokens
        results[0][2:],                   # memory
        model.attn_map                    # attention map
    )


def get_returns_ManiSkill(env, model, ret, seed, episode_timeout, context_length, device, act_dim, config, mean, std, 
                          use_argmax=False, create_video=False, sparse_reward=False):
    # env = None
    # device = torch.device("cpu")
    model.to(device)

    set_seed(seed)

    # eval_output_dir = f"ManiSkill/val_videos/{config['model_mode']}/{config['text_description']}"
    # env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos", render_mode="all", sim_backend="cpu")

    # env = gym.make("PushCube-v1", num_envs=1, **env_kwargs) 
    # env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    # if isinstance(env.action_space, gym.spaces.Dict):
    #     env = FlattenActionSpaceWrapper(env) 
    # env = RecordEpisode(env, output_dir=eval_output_dir, save_trajectory=False, trajectory_name=f"rate_val", max_steps_per_video=50, video_fps=30)
    # env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    scale = 1
    
    # Reset environment and get initial state
    state_0, _ = env.reset(seed=seed)
    state_0 = state_0['rgb'][0]
    
    # Convert state to expected format [C, H, W] and add batch/sequence dimensions
    state = state_0.float().permute(2, 0, 1).to(device)  # [C, H, W]
    state = state.unsqueeze(0).unsqueeze(0)   # [1, 1, C, H, W]
    
    # Initialize episode tracking
    episode_return = 0
    episode_length = 0
    done = False
    HISTORY_LEN = context_length
    
    # Initialize state/action tracking
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    
    # Initialize return targets and timesteps
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    
    # Initialize memory tracking
    mem_tokens = (model.mem_tokens.repeat(1, 1, 1).detach() 
                    if model.mem_tokens is not None else None)
    saved_context = None
    segment = 0
    prompt_steps = 0
    
    # Initialize output tracking
    out_states = [state.cpu().numpy()]
    frames, rews, act_list, memories, eval_metrics = [env.render()], [], [], [], defaultdict(list)
    
    for t in range(episode_timeout):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        # RATE / RMT / TrXL
        if config["model_mode"] not in ['DT', 'DTXL']:
            # For non-DT models, truncate sequences when they exceed history length
            if actions.shape[0] > HISTORY_LEN:
                segment += 1

                keep_steps = prompt_steps if prompt_steps > 0 else 1

                actions = actions[-keep_steps:, :]
                states = states[:, -keep_steps:, :, :, :]
                target_return = target_return[:, -keep_steps:]
                timesteps = timesteps[:, -keep_steps:]
                
                # Update memory tokens periodically
                if t % context_length == 0 and t > 5:
                    mem_tokens = new_mem
                    saved_context = new_notes
                    
                    if create_video:
                        memory_norm = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {memory_norm}')
                        
        # DT / DTXL          
        else:
            if actions.shape[0] > HISTORY_LEN:
                segment += 1
                
                keep_steps = prompt_steps if prompt_steps > 0 else 1

                actions = actions[-keep_steps:, :]
                states = states[:, -keep_steps:, :, :, :]
                target_return = target_return[:, -keep_steps:]
                timesteps = timesteps[:, -keep_steps:]
                    
                # Update memory tokens periodically
                if t % context_length == 0 and t > 5:
                    if create_video:
                        memory_norm = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {memory_norm}')
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
        # print(states_norm.device, act_to_pass.device if act_to_pass is not None else None, timesteps.device)
        try:
            sampled_action, new_mem, new_notes, attn_map = sample(
                model=model,
                x=states_norm,
                block_size=HISTORY_LEN,
                steps=1,
                sample=True,
                actions=act_to_pass,
                rtgs=target_return.unsqueeze(-1),
                timestep=timesteps,
                mem_tokens=mem_tokens,
                saved_context=saved_context
            )
        except Exception as e:
            print(f"Error during sampling at timestep {t}:")
            print(f"states shape: {states.shape}")
            print(f"target_return shape: {target_return.shape}") 
            print(f"actions shape: {act_to_pass.shape if act_to_pass is not None else None}")
            print(f"Exception: {str(e)}")
            raise
            
        if new_mem is not None:
            memories.append(mem_tokens.detach().cpu().numpy())

        act = sampled_action#.detach().cpu().numpy()
        
        actions[-1, :] = act
        act_list.append(act)
        # print(act.device)
        state, reward, terminated, truncated, eval_infos = env.step(act) # state [H, W, C], need [C, H, W]
        if sparse_reward:
            reward = (reward==1).to(torch.float32)
        # print(reward.device, terminated.device, truncated.device)
        done = torch.logical_or(terminated, truncated).item()
        if "final_info" in eval_infos:
            for k, v in eval_infos["final_info"]["episode"].items():
                eval_metrics[k].append(v)
        state = state['rgb'][0]
        frames.append(env.render())
        state = state.float().permute(2, 0, 1).to(device)
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
            break  
        
    if create_video == True:
        print("\n")

    # env._env.close()
    # env.close()
    # torch.cuda.empty_cache()

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