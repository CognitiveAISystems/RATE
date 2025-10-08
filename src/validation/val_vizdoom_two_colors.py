import torch
import numpy as np

from src.envs.vizdoom_two_colors import env_vizdoom2, doom_environment2
from src.utils.set_seed import set_seed

env_args = {
    'simulator':'doom', 
    'scenario':'custom_scenario{:003}.cfg', # custom_scenario{:003}.cfg
    'test_scenario':'', 
    'screen_size':'320X180', 
    'screen_height':64, 
    'screen_width':112, 
    'num_environments':16,# 16
    'limit_actions':True, 
    'scenario_dir':'src/envs/vizdoom_two_colors/scenarios/', 
    'test_scenario_dir':'', 
    'show_window':False, 
    'resize':True, 
    'multimaze':True, 
    'num_mazes_train':16, 
    'num_mazes_test':1, # 64 
    'disable_head_bob':False, 
    'use_shaping':False, 
    'fixed_scenario':False, 
    'use_pipes':False, 
    'num_actions':0, 
    'hidden_size':128, 
    'reload_model':'', 
    'model_checkpoint':'../3dcdrl/saved_models/two_col_p1_checkpoint_0198658048.pth.tar',
    'conv1_size':16, 
    'conv2_size':32, 
    'conv3_size':16, 
    'learning_rate':0.0007, 
    'momentum':0.0, 
    'gamma':0.99, 
    'frame_skip':4, 
    'train_freq':4, 
    'train_report_freq':100, 
    'max_iters':5000000, 
    'eval_freq':1000, 
    'eval_games':50, 
    'model_save_rate':1000, 
    'eps':1e-05, 
    'alpha':0.99, 
    'use_gae':False, 
    'tau':0.95, 
    'entropy_coef':0.001, 
    'value_loss_coef':0.5, 
    'max_grad_norm':0.5, 
    'num_steps':128, 
    'num_stack':1, 
    'num_frames':200000000, 
    'use_em_loss':False, 
    'skip_eval':False, 
    'stoc_evals':False, 
    'model_dir':'', 
    'out_dir':'./', 
    'log_interval':100, 
    'job_id':12345, 
    'test_name':'test_000', 
    'use_visdom':False, 
    'visdom_port':8097, 
    'visdom_ip':'http://10.0.0.1'                 
}

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

def get_returns_VizDoom(model, ret, seed, episode_timeout, context_length, device, config, use_argmax=False, create_video=False):
    
    set_seed(seed)
    max_ep_len = episode_timeout
        
    scene = 0
    scenario = env_args['scenario_dir'] + env_args['scenario'].format(scene)
    config_env = scenario

    env = env_vizdoom2.DoomEnvironmentDisappear(
        scenario=config_env,
        show_window=False,
        use_info=True,
        use_shaping=False, #if False bonus reward if #shaping reward is always: +1,-1 in two_towers
        frame_skip=2,
        no_backward_movement=True,
        seed=seed
    )
    
    # Get dtype from config
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(config["dtype"], torch.float32)  # default to float32 if not specified

    state0 = env.reset()
    state = torch.tensor(state0['image']).float()
    state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
    
    rews = []
    act_dim = 1
    states = state.to(device=device, dtype=dtype)
    actions = torch.zeros((0, act_dim), device=device, dtype=dtype)
    rewards = torch.zeros(0, device=device, dtype=dtype)
    target_return = torch.tensor(ret, device=device, dtype=dtype).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    is_lstm = hasattr(model, 'backbone') and model.backbone in ['lstm', 'gru']

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if hasattr(model, 'mem_tokens') and model.mem_tokens is not None else None
    saved_context = None
    hidden = model.reset_hidden(1, device) if is_lstm else None
    memory_states = model.init_memory(1, device) if config["model_mode"] == "ELMUR" else None

    # Initialize variables that will be used in the loop
    new_mem_tokens = mem_tokens
    new_context = saved_context
    new_memory_states = memory_states

    episode_return, episode_length = 0, 0
    
    # Debug: Track actions and rewards
    debug_actions = []
    debug_rewards = []
    
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
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

        action_probs = torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy()

        if not use_argmax:
            act = np.random.choice([0, 1, 2, 3, 4], p=action_probs)
        else:
            act = np.argmax(action_probs)

        actions[-1] = act
        
        # Debug: Track action probabilities and chosen action
        if t < 10:  # Only track first 10 steps to avoid spam
            debug_actions.append((t, action_probs, act))
        
        state, reward, done, info = env.step(act)
        state = np.float32(state['image'])
        state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
        rews.append(reward)
        cur_state = torch.from_numpy(state).to(device=device).float()
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = reward
        pred_return = target_return[0,-1] - reward
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (1)], dim=1)
        episode_return += reward
        episode_length += 1
        
        # Debug: Track rewards
        debug_rewards.append((t, reward, episode_return))
        
        if done:
            break  
        
    if create_video == True:
        print("\n")
    
    # Debug: Print debug information
    print(f"Debug info for seed {seed}:")
    print(f"  Episode return: {episode_return}")
    print(f"  Episode length: {episode_length}")
    print(f"  First 10 actions: {[a[2] for a in debug_actions[:10]]}")
    print(f"  First 10 action probs: {[a[1] for a in debug_actions[:10]]}")
    print(f"  First 10 rewards: {[r[1] for r in debug_rewards[:10]]}")
    
    env.close()
    return episode_return, t