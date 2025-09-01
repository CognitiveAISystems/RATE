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
def sample(model, x, block_size, steps, sample=False, top_k=None, actions=None, rtgs=None, timestep=None, mem_tokens=1, saved_context=None, hidden=None):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        
        if saved_context is not None:
            results = model(x_cond, actions, rtgs,None, timestep, *saved_context, mem_tokens=mem_tokens, hidden=hidden)
        else:
            results = model(x_cond, actions, rtgs,None, timestep, mem_tokens=mem_tokens, hidden=hidden) 

        logits = results['logits'][:,-1,:]
        memory = results.get('new_mems', None)
        mem_tokens = results.get('mem_tokens', None)
        hidden = results.get('hidden', None)
        attn_map = getattr(model, 'attn_map', None)
        
    return logits, mem_tokens, memory, attn_map, hidden

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
    
    state0 = env.reset()
    state = torch.tensor(state0['image']).float()
    state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
    
    rews = []
    act_dim = 1
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    is_lstm = hasattr(model, 'backbone') and model.backbone in ['lstm', 'gru']

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    saved_context = None
    hidden = model.reset_hidden(1, device) if is_lstm else None

    episode_return, episode_length = 0, 0
    
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        if not is_lstm and actions.shape[0] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[slice_index:] if slice_index == 1 else actions[slice_index:,:]
            states = states[:, slice_index:, :]
            target_return = target_return[:,slice_index:]
            timesteps = timesteps[:, slice_index:]
            if t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context

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
            hidden=hidden
        )

        sampled_action, new_mem_tokens, new_context, attn_map, new_hidden = sample_outputs

        if is_lstm:
            hidden = new_hidden

        action_probs = torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy()

        if not use_argmax:
            act = np.random.choice([0, 1, 2, 3, 4], p=action_probs)
        else:
            act = np.argmax(action_probs)

        actions[-1] = act
        
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
        
        if done:
            break  
        
    if create_video == True:
        print("\n")
    
    env.close()
    return episode_return, t