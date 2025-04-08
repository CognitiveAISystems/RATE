import torch
import numpy as np
import popgym
import gymnasium as gym

from src.additional.gen_popgym_data.get_dataset_collectors_ckpt import env_names_dict
from src.utils.additional_data_processors import coords_to_idx, idx_to_coords
from src.utils.set_seed import set_seed



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
            
        # logits = results[0][0].detach()[:,-1,:]
        # mem_tokens = results[1]
        # memory = results[0][2:]

        logits = results['logits'][:,-1,:]
        memory = results['new_mems']
        mem_tokens = results['mem_tokens']

        attn_map = model.attn_map
        
    return logits, mem_tokens, memory, attn_map

def get_returns_POPGym(model, ret, seed, episode_timeout, context_length, device, config, use_argmax=False):
    def get_env_class(env_name):
        return next((value[1] for value in env_names_dict.values() if value[0].startswith(env_name)), None)
    
    set_seed(seed)
    env_name = config['model']['env_name'] + '-v0'
    env = get_env_class(env_name)()
    states, _ = env.reset(seed=seed)
    states = torch.tensor(states, dtype=torch.float32, device=device).reshape(1, 1, -1)
    actions = torch.zeros((0, 1), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    episode_return, episode_length = 0, 0
    make_np_array = False

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    saved_context = None

    for t in range(episode_timeout):
        actions = torch.cat([actions, torch.zeros((1, 1), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        if actions.shape[0] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[slice_index:] if slice_index == 1 else actions[slice_index:,:]
            states = states[:, slice_index:, :]
            target_return = target_return[:,slice_index:]
            timesteps = timesteps[:, slice_index:]

            if t % context_length == 0:
                mem_tokens = new_mem
                saved_context = new_notes

        act_to_pass = None if t == 0 else actions.unsqueeze(0)[:, 1:, :] 
        if act_to_pass is not None and act_to_pass.shape[1] == 0:
            act_to_pass = None

        
        sample_outputs = sample(
            model=model,
            x=states,
            block_size=context_length,
            steps=1,
            sample=True,
            actions=act_to_pass,
            rtgs=target_return.unsqueeze(-1),
            timestep=timesteps,
            mem_tokens=mem_tokens,
            saved_context=saved_context
        )
        
        sampled_action, new_mem, new_notes, attn_map = sample_outputs
        
        action_probs = torch.softmax(sampled_action, dim=-1).squeeze().cpu().numpy()
        
        if isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            if any(game in env_name for game in ['Battleship', 'MineSweeper']):
                act = action_probs.argmax()
                act = idx_to_coords(act, board_size=8 if 'Battleship' in env_name else 4)
            else:
                if not use_argmax:
                    act = np.random.choice(env.action_space.n, p=action_probs)
                else:
                    act = action_probs.argmax()
        else:
            # NoisyPositionOnlyCartPoleMedium
            act = sampled_action.item()
            act = np.array([act])
            # normalized_action = sampled_action.squeeze().cpu().numpy()
            # act = normalized_action * (env.action_space.high - env.action_space.low) / 2 + \
            #       (env.action_space.high + env.action_space.low) / 2
        
        actions[-1] = act if isinstance(act, (int, float)) else act[0]
        state, reward, truncated, terminated, _ = env.step(act)
        done = truncated or terminated
        state = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, 1, -1)
        states = torch.cat([states, state], dim=1)
        rewards[-1] = reward
        pred_return = target_return[:, -1] - reward
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (1)], dim=1)
        episode_return += reward
        episode_length += 1
        
        if done:
            break  
        
    env.close()

    return episode_return, None, t, None, None, attn_map, None