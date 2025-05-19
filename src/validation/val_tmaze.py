import numpy as np
from tqdm import tqdm
from src.envs.tmaze.tmaze import TMazeClassicPassive
import torch
from src.utils.set_seed import set_seed, seeds_list


@torch.no_grad()
def sample(model, x, block_size, steps, sample=False, top_k=None, actions=None, rtgs=None, timestep=None, mem_tokens=1, saved_context=None, hidden=None):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        
        if saved_context is not None:
            results = model(x_cond, actions, rtgs, None, timestep, *saved_context, mem_tokens=mem_tokens, hidden=hidden)
        else:
            results = model(x_cond, actions, rtgs, None, timestep, mem_tokens=mem_tokens, hidden=hidden)

        logits = results['logits'][:,-1,:]
        memory = results.get('new_mems', None)
        mem_tokens = results.get('mem_tokens', None)
        hidden = results.get('hidden', None)
        attn_map = getattr(model, 'attn_map', None)
        
    return logits, mem_tokens, memory, attn_map, hidden

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
    states = torch.tensor(states).reshape(batch_size, 1, channels).to(device=device, dtype=torch.float32)

    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    rewards = torch.zeros(batch_size, device=device)
    episode_rewards = torch.zeros(batch_size, device=device)
    successes = 0

    actions = torch.zeros((batch_size, 0, 1), device=device, dtype=torch.float32)
    rewards_hist = torch.zeros((batch_size, 0), device=device, dtype=torch.float32)
    target_return = torch.full((batch_size, 1, 1), ret, device=device, dtype=torch.float32)
    timesteps = torch.zeros((batch_size, 1, 1), device=device, dtype=torch.long)
    
    is_lstm = hasattr(model, 'backbone') and model.backbone in ['lstm', 'gru']
    
    mem_tokens = model.mem_tokens.repeat(1, batch_size, 1).detach() if hasattr(model, 'mem_tokens') and model.mem_tokens is not None else None
    saved_context = None
    hidden = model.reset_hidden(batch_size, device) if is_lstm else None

    for t in tqdm(range(max_ep_len), desc=f"Steps (T={max_ep_len}, corridor={corridor_length}, mode={config['model_mode']})"):
        # Prepare actions and rewards for this step
        actions = torch.cat([actions, torch.zeros((batch_size, 1, 1), device=device)], dim=1)
        rewards_hist = torch.cat([rewards_hist, torch.zeros((batch_size, 1), device=device)], dim=1)

        if not is_lstm and actions.shape[1] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[:, slice_index:]
            states = states[:, slice_index:]
            target_return = target_return[:, slice_index:]
            timesteps = timesteps[:, slice_index:]
            if t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context

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
            hidden=hidden
        )

        sampled_action, new_mem_tokens, new_context, attn_map, new_hidden = sample_outputs
        
        if is_lstm:
            hidden = new_hidden

        act = torch.argmax(torch.softmax(sampled_action, dim=-1).squeeze(), dim=-1)  # (batch,)
        actions[:, -1, 0] = act

        next_states = []
        for i, env in enumerate(envs):
            if done[i]:
                next_states.append(states[i, -1].cpu().numpy())
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
        next_states = torch.tensor(next_states).reshape(batch_size, 1, channels).to(device=device, dtype=torch.float32)
        states = torch.cat([states, next_states], dim=1)
        rewards_hist[:, -1] = rewards
        
        timesteps = torch.cat([timesteps, torch.ones((batch_size, 1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        # Update target_return
        pred_return = target_return[:, -1, 0] - (rewards / scale)
        pred_return = pred_return.reshape(batch_size, 1, 1)
        target_return = torch.cat([target_return, pred_return], dim=1)

    # Success: reward == 1.0
    successes = (episode_rewards == 1.0).sum().item()
    return episode_rewards.cpu().tolist(), successes