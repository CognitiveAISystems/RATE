import numpy as np
from tqdm import tqdm
from src.envs.tmaze.tmaze import TMazeClassicPassive
import torch
from src.utils.set_seed import set_seed, seeds_list


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

def get_returns_TMaze(model, ret, seeds, episode_timeout, corridor_length, context_length, device, config, create_video=False):
    set_seed(42)
    batch_size = len(seeds)
    scale = 1
    channels = 5
    hint_steps = 1
    max_ep_len = episode_timeout

    envs = [TMazeClassicPassive(episode_length=episode_timeout, corridor_length=corridor_length, penalty=0, seed=seeds[i], goal_reward=ret) for i in range(batch_size)]
    states = []
    mem_states = []
    mem_state2s = []
    for env in envs:
        state = env.reset()
        mem_states.append(state[2])
        mem_state2s.append(state)
        state = np.concatenate((state, np.array([0])))
        state = np.concatenate((state, np.array([np.random.randint(low=-1, high=1+1)])))
        states.append(state)
    states = np.stack(states)  # (batch, channels)
    # Get dtype from config
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(config["dtype"], torch.float32)  # default to float32 if not specified

    states = torch.tensor(states).reshape(batch_size, 1, channels).to(device=device, dtype=dtype)

    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    rewards = torch.zeros(batch_size, device=device, dtype=dtype)
    episode_rewards = torch.zeros(batch_size, device=device, dtype=dtype)
    successes = 0

    actions = torch.zeros((batch_size, 0, 1), device=device, dtype=dtype)
    rewards_hist = torch.zeros((batch_size, 0), device=device, dtype=dtype)
    target_return = torch.full((batch_size, 1, 1), ret, device=device, dtype=dtype)
    timesteps = torch.zeros((batch_size, 1, 1), device=device, dtype=torch.long)
    
    is_lstm = hasattr(model, 'backbone') and model.backbone in ['lstm', 'gru']
    
    mem_tokens = model.mem_tokens.repeat(1, batch_size, 1).detach() if hasattr(model, 'mem_tokens') and model.mem_tokens is not None else None
    saved_context = None
    hidden = model.reset_hidden(batch_size, device) if is_lstm else None
    memory_states = model.init_memory(batch_size, device) if config["model_mode"] == "MATL" else None

    for t in tqdm(range(max_ep_len), desc=f"Steps (T={max_ep_len}, corridor={corridor_length}, mode={config['model_mode']})"):
        # Prepare actions and rewards for this step
        actions = torch.cat([actions, torch.zeros((batch_size, 1, 1), device=device, dtype=dtype)], dim=1)
        rewards_hist = torch.cat([rewards_hist, torch.zeros((batch_size, 1), device=device, dtype=dtype)], dim=1)

        if not is_lstm and actions.shape[1] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[:, slice_index:]
            states = states[:, slice_index:]
            target_return = target_return[:, slice_index:]
            timesteps = timesteps[:, slice_index:]
            if t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context
                if config["model_mode"] == "MATL":
                    memory_states = new_memory_states

        if is_lstm:
            states_to_pass = states[:, -1:, 1:]
            act_to_pass = None if t == 0 else actions[:, -1:, :]
            rtg_to_pass = target_return[:, -1:, :]
            timesteps = timesteps[:, -1:]
        else:
            states_to_pass = states[:, :, 1:]
            act_to_pass = None if t == 0 else actions[:, 1:, :]
            rtg_to_pass = target_return
            if act_to_pass is not None and act_to_pass.shape[1] == 0:
                act_to_pass = None

        # For MATL we use the segment approach as during training
        if config["model_mode"] == "MATL":
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
            timestep=timesteps.squeeze(-1),
            mem_tokens=mem_tokens,
            saved_context=saved_context,
            hidden=hidden,
            memory_states=memory_states,
            pos_offset=pos_offset_val
        )

        sampled_action, new_mem_tokens, new_context, attn_map, new_hidden, new_memory_states = sample_outputs
        
        if is_lstm:
            hidden = new_hidden

        act = torch.argmax(torch.softmax(sampled_action, dim=-1).squeeze(), dim=-1)  # (batch,)
        actions[:, -1, 0] = act

        next_states = []
        for i, env in enumerate(envs):
            if done[i]:
                # Convert to float32 before numpy conversion (bfloat16 not supported by numpy)
                state_np = states[i, -1].cpu().float().numpy()
                next_states.append(state_np)
                continue
            state, reward, d, info = env.step(int(act[i].item()))
            if t < hint_steps-1:
                state[2] = mem_state2s[i][2]
            # {x, y, hint} -> {x, y, hint, flag}
            if state[0] != env.corridor_length:
                state = np.concatenate((state, np.array([0])))
            else:
                state = np.concatenate((state, np.array([1])))

            # {x, y, hint, flag} -> {x, y, hint, flag, noise}  
            state = np.concatenate((state, np.array([np.random.randint(low=-1, high=1+1)])))
            next_states.append(state)
            rewards[i] = reward
            episode_rewards[i] += reward
            done[i] = d
        next_states = np.stack(next_states)
        next_states = torch.tensor(next_states).reshape(batch_size, 1, channels).to(device=device, dtype=dtype)
        states = torch.cat([states, next_states], dim=1)
        rewards_hist[:, -1] = rewards
        
        timesteps = torch.cat([timesteps, torch.ones((batch_size, 1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        # Update target_return
        pred_return = target_return[:, -1, 0] - (rewards / scale)
        pred_return = pred_return.reshape(batch_size, 1, 1)
        target_return = torch.cat([target_return, pred_return], dim=1)

    # Success: reward == 1.0
    successes = (episode_rewards == 1.0).sum().item()
    # Convert to float32 before converting to list (for bfloat16 compatibility)
    return episode_rewards.cpu().float().tolist(), successes