import torch
import numpy as np
import popgym
import gymnasium as gym

from src.additional.gen_popgym_data.get_dataset_collectors_ckpt import env_names_dict
from src.utils.additional_data_processors import coords_to_idx, idx_to_coords
from src.utils.set_seed import set_seed


@torch.no_grad()
def sample(
    model, x, block_size, steps, sample=False, top_k=None, actions=None, 
    rtgs=None, timestep=None, mem_tokens=1, saved_context=None, hidden=None,
    memory_states=None, pos_offset=0
):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        
        if saved_context is not None:
            results = model(
                x_cond, actions, rtgs, None, timestep, *saved_context, mem_tokens=mem_tokens, 
                hidden=hidden, memory_states=memory_states, pos_offset=pos_offset)
        else:
            results = model(
                x_cond, actions, rtgs, None, timestep, mem_tokens=mem_tokens, 
                hidden=hidden, memory_states=memory_states, pos_offset=pos_offset)

        logits = results['logits'][:,-1,:]
        memory = results.get('new_mems', None)
        mem_tokens = results.get('mem_tokens', None)
        hidden = results.get('hidden', None)
        attn_map = getattr(model, 'attn_map', None)
        memory_states = results.get('memory_states', None)
        
    return logits, mem_tokens, memory, attn_map, hidden, memory_states

def get_returns_POPGym(model, ret, seed, episode_timeout, context_length, device, config, use_argmax=False):
    def get_env_class(env_name):
        return next((value[1] for value in env_names_dict.values() if value[0].startswith(env_name)), None)
    
        # Get dtype from config
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(config["dtype"], torch.float32)  # default to float32 if not specified

    set_seed(seed)
    env_name = config['model']['env_name'] + '-v0'
    env = get_env_class(env_name)()
    states, _ = env.reset(seed=seed)
    states = torch.tensor(states, dtype=dtype, device=device).reshape(1, 1, -1)
    actions = torch.zeros((0, 1), device=device, dtype=dtype)
    rewards = torch.zeros(0, device=device, dtype=dtype)
    target_return = torch.tensor(ret, device=device, dtype=dtype).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    episode_return, episode_length = 0, 0

    is_lstm = hasattr(model, 'backbone') and model.backbone in ['lstm', 'gru']

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if hasattr(model, 'mem_tokens') and model.mem_tokens is not None else None
    saved_context = None
    hidden = model.reset_hidden(1, device) if is_lstm else None
    memory_states = model.init_memory(1, device) if config["model_mode"] == "ELMUR" else None

    # Initialize variables that will be used in the loop
    new_mem_tokens = mem_tokens
    new_context = saved_context
    new_memory_states = memory_states

    for t in range(episode_timeout):
        actions = torch.cat([actions, torch.zeros((1, 1), device=device, dtype=dtype)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device, dtype=dtype)])
        
        if not is_lstm and actions.shape[0] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[slice_index:] if slice_index == 1 else actions[slice_index:,:]
            states = states[:, slice_index:, :]
            target_return = target_return[:,slice_index:]
            timesteps = timesteps[:, slice_index:]
            if t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context
                if config["model_mode"] == "ELMUR":
                    memory_states = new_memory_states

        if is_lstm:
            states_to_pass = states[:, -1:, :]
            act_to_pass = None if t == 0 else actions[-1:].unsqueeze(0)
            rtg_to_pass = target_return[:, -1:].unsqueeze(-1)
            time_to_pass = timesteps[:, -1:]
        else:
            states_to_pass = states
            act_to_pass = None if t == 0 else actions.unsqueeze(0)[:, 1:, :]
            rtg_to_pass = target_return.unsqueeze(-1)
            time_to_pass = timesteps
            if act_to_pass is not None and act_to_pass.shape[1] == 0:
                act_to_pass = None

        # For ELMUR we use the segment approach as during training
        if config["model_mode"] == "ELMUR":
            # Number of the current segment and position inside the segment
            segment_idx = t // context_length
            pos_in_segment = t % context_length
            # pos_offset corresponds to the beginning of the current segment
            # Get sequence format multiplier
            sequence_format = getattr(model, 'sequence_format', 'sra')
            multiplier = model.get_sequence_length_multiplier()
            pos_offset_val = segment_idx * context_length * multiplier
        else:
            window_len = min(context_length, t + 1)
            pos_offset_val = (t - window_len + 1) * 3
        
        sample_outputs = sample(
            model=model,
            x=states_to_pass,
            block_size=context_length,
            steps=1,
            sample=True,
            actions=act_to_pass,
            rtgs=rtg_to_pass,
            timestep=time_to_pass,
            mem_tokens=mem_tokens,
            saved_context=saved_context,
            hidden=hidden,
            memory_states=memory_states,
            pos_offset=pos_offset_val
        )
        
        sampled_action, new_mem_tokens, new_context, attn_map, new_hidden, new_memory_states = sample_outputs
        
        if is_lstm:
            hidden = new_hidden
        
        action_probs = torch.softmax(sampled_action, dim=-1).squeeze().cpu().numpy()
        
        if isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            if any(game in env_name for game in ['Battleship', 'MineSweeper']):
                act = action_probs.argmax()
                if 'BattleshipEasy' in env_name:
                    act = idx_to_coords(act, board_size=8)
                elif 'BattleshipMedium' in env_name:
                    act = idx_to_coords(act, board_size=10)
                elif 'BattleshipHard' in env_name:
                    act = idx_to_coords(act, board_size=12)
                elif 'MineSweeperEasy' in env_name:
                    act = idx_to_coords(act, board_size=4)
                elif 'MineSweeperMedium' in env_name:
                    act = idx_to_coords(act, board_size=6)
                elif 'MineSweeperHard' in env_name:
                    act = idx_to_coords(act, board_size=8)
            else:
                if not use_argmax:
                    act = np.random.choice(env.action_space.n, p=action_probs)
                else:
                    act = action_probs.argmax()
        else:
            act = sampled_action.item()
            act = np.array([act])
        
        actions[-1] = act if isinstance(act, (int, float)) else act[0]
        state, reward, truncated, terminated, _ = env.step(act)
        done = truncated or terminated
        state = torch.tensor(state, dtype=dtype, device=device).reshape(1, 1, -1)
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