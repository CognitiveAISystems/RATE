import torch
import numpy as np
from src.utils.set_seed import set_seed
from collections import defaultdict




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


def get_returns_MIKASARobo(
    env, model, ret, seed, episode_timeout, context_length, device, config, 
    use_argmax=False, create_video=False
):

    set_seed(seed)
    scale = 1

    envs_num = 20

    # model = model.cpu()
    # device = torch.device('cpu')
# Get dtype from config
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(config["dtype"], torch.float32)  # default to float32 if not specified
    
    # Reset environment and get initial state
    state_0, _ = env.reset(seed=seed)
    state_0 = state_0['rgb'] # envs_numx128x128x6
    
    # Convert state to expected format [C, H, W] and add batch/sequence dimensions
    state = state_0.float().permute(0, 3, 1, 2).to(device)  # envs_numx6x128x128
    state = state.unsqueeze(1)   # envs_numx1x6x128x128
    
    # Initialize episode tracking
    episode_return = torch.zeros((envs_num), device=device, dtype=dtype)
    episode_length = 0
    done = False
    HISTORY_LEN = context_length
    
    # Initialize state/action tracking
    states = state.to(device=device, dtype=dtype)
    actions = torch.zeros((envs_num, 0, config['model']['act_dim']), device=device, dtype=dtype)
    
    # Initialize return targets and timesteps
    target_return = torch.ones((envs_num, 1), device=device, dtype=dtype) * ret
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    is_lstm = hasattr(model, 'backbone') and model.backbone in ['lstm', 'gru']
    
    # Initialize memory tracking
    mem_tokens = (model.mem_tokens.repeat(1, envs_num, 1).detach() if model.mem_tokens is not None else None)
    saved_context = None
    hidden = model.reset_hidden(envs_num, device) if is_lstm else None
    memory_states = model.init_memory(envs_num, device) if config["model_mode"] == "ELMUR" else None

    # Initialize variables that will be used in the loop
    new_mem_tokens = mem_tokens
    new_context = saved_context
    new_memory_states = memory_states

    segment = 0
    prompt_steps = 0
    
    frames, rews, act_list, memories, eval_metrics = [env.render()], [], [], [], defaultdict(list)
    
    for t in range(episode_timeout):
        actions = torch.cat([actions, torch.zeros((envs_num, 1, config['model']['act_dim']), device=device)], dim=1)
        
        if not is_lstm and actions.shape[1] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[:, slice_index:]
            states = states[:, slice_index:]
            target_return = target_return[:, slice_index:]
            timesteps = timesteps[:, slice_index:]
            if t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context
                if config["model_mode"] == "ELMUR":
                    memory_states = new_memory_states

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
            x=states_norm,
            block_size=context_length, 
            steps=1, 
            sample=True, 
            actions=act_to_pass, 
            rtgs=target_return.unsqueeze(-1), 
            timestep=timesteps, 
            mem_tokens=mem_tokens,
            saved_context=saved_context,
            hidden=hidden,
            memory_states=memory_states,
            pos_offset=pos_offset_val
        )

        sampled_action, new_mem_tokens, new_context, attn_map, new_hidden, new_memory_states = sample_outputs

        if is_lstm:
            hidden = new_hidden
            
        act = sampled_action#.detach().cpu().numpy()
        act = act # envs_numx1x8
        
        actions[:, -1, :] = act
        act_list.append(act)
        
        state, reward, terminated, truncated, eval_infos = env.step(act) # state [H, W, C], need [C, H, W]

        # * we work in discrete reward setting (for us reward is success_once)
        reward = eval_infos['success'].float()

        done = torch.logical_or(terminated, truncated)

        if "final_info" in eval_infos:
            for k, v in eval_infos["final_info"]["episode"].items():
                # v = v.float().mean().item()
                eval_metrics[k].append(v)

        state = state['rgb'].float().permute(0, 3, 1, 2).to(device).to(dtype)
        state = state.unsqueeze(1)   # envs_numx1x6x128x128
        
        cur_state = state.to(device).to(dtype)
        states = torch.cat([states, cur_state], dim=1)

        pred_return = target_return[:,-1] - reward

        target_return = torch.cat([target_return, pred_return.unsqueeze(-1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (1)], dim=1)
        episode_return += reward
        episode_length += 1

        # print(t, reward, ret, episode_return, target_return[:, -1].item())
        
    if create_video == True:
        print("\n")

    episode_return = episode_return.mean().item()

    return episode_return, act_list, t, None, memories, attn_map, frames, eval_metrics