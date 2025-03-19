import torch
import numpy as np
from src.utils.set_seed import set_seed
from collections import defaultdict




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


def get_returns_MIKASARobo(env, model, ret, seed, episode_timeout, context_length, device, config, 
                           use_argmax=False, create_video=False):

    set_seed(seed)
    scale = 1

    # model = model.cpu()
    # device = torch.device('cpu')
    
    # Reset environment and get initial state
    state_0, _ = env.reset(seed=seed)
    state_0 = state_0['rgb'] # 16x128x128x6
    
    # Convert state to expected format [C, H, W] and add batch/sequence dimensions
    state = state_0.float().permute(0, 3, 1, 2).to(device)  # 16x6x128x128
    state = state.unsqueeze(1)   # 16x1x6x128x128
    
    # Initialize episode tracking
    episode_return = torch.zeros((16), device=device, dtype=torch.float32)
    episode_length = 0
    done = False
    HISTORY_LEN = context_length
    
    # Initialize state/action tracking
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((16, 0, config['model']['act_dim']), device=device, dtype=torch.float32)
    
    # Initialize return targets and timesteps
    target_return = torch.ones((16, 1), device=device, dtype=torch.float32) * ret
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    
    # Initialize memory tracking
    mem_tokens = (model.mem_tokens.repeat(1, 16, 1).detach() if model.mem_tokens is not None else None)
    saved_context = None
    segment = 0
    prompt_steps = 0
    
    frames, rews, act_list, memories, eval_metrics = [env.render()], [], [], [], defaultdict(list)
    
    for t in range(episode_timeout):
        actions = torch.cat([actions, torch.zeros((16, 1, config['model']['act_dim']), device=device)], dim=1)
        
        # RATE / RMT / TrXL
        if config["model_mode"] not in ['DT', 'DTXL']:
            # For non-DT models, truncate sequences when they exceed history length
            if actions.shape[0] > HISTORY_LEN:
                segment += 1

                keep_steps = prompt_steps if prompt_steps > 0 else 1

                actions = actions[:, -keep_steps:, :]
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

                actions = actions[:, -keep_steps:, :]
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
            act_to_pass = actions[:, 1:, :]
            if act_to_pass.shape[1] == 0:
                act_to_pass = None 
        
        states_norm = states / 255.0
        prt = act_to_pass.shape if act_to_pass is not None else None
        mem_tokens_ = mem_tokens.shape if mem_tokens is not None else None
        saved_context_ = saved_context[0].shape if saved_context is not None else None
        # print(f"\n{states_norm.shape=} | {prt} | {target_return.shape=} | {timesteps.shape=} | {mem_tokens_} | {saved_context_}")
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
            
        if new_mem is not None:
            memories.append(mem_tokens.detach().cpu().numpy())

        act = sampled_action#.detach().cpu().numpy()
        act = act # 16x1x8
        
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

        state = state['rgb'].float().permute(0, 3, 1, 2).to(device)
        state = state.unsqueeze(1)   # 16x1x6x128x128
        
        cur_state = state.to(device)
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