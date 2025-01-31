import torch
import numpy as np

from VizDoom.VizDoom_src.utils import z_normalize, inverse_z_normalize
from VizDoom.VizDoom_src.utils import env_vizdoom2
from TMaze_new.TMaze_new_src.utils import set_seed

env_args = {
    'simulator':'doom', 
    'scenario':'custom_scenario{:003}.cfg', #custom_scenario{:003}.cfg
    'test_scenario':'', 
    'screen_size':'320X180', 
    'screen_height':64, 
    'screen_width':112, 
    'num_environments':16,# 16
    'limit_actions':True, 
    'scenario_dir':'VizDoom/VizDoom_src/env/', 
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

def get_returns_VizDoom(model, ret, seed, episode_timeout, context_length, device, act_dim, config, mean, std, use_argmax=False, create_video=False):
    
    set_seed(seed)

    scale = 1
    max_ep_len = episode_timeout#* 3

        
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
        seed=seed)
    
    state = torch.tensor(env.reset()['image']).float()
    state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
   
    out_states = []
    out_states.append(state.cpu().numpy())
    done = True
    HISTORY_LEN = context_length 
    
    rews = []
    act_dim = 1
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    episode_return, episode_length = 0, 0


    # print(device, states.device)
    segment = 0
    prompt_steps = 0
    
    act_list= []
    memories = []
    
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

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
                
        if config["data_config"]["normalize"] == 2:
            b, l, c, h, w = states.shape
            states_norm = states.reshape(b*l, c, h, w)
            states_norm = z_normalize(states_norm, mean.to(device), std.to(device))
            states_norm = states_norm.reshape(b, l, c, h, w).to(device)
        elif config["data_config"]["normalize"] == 1:
            states_norm = states / 255.0
        elif config["data_config"]["normalize"] == 0:
            states_norm = states
        
        # print(f"{t=}, {states_norm.shape=}, {actions.shape=}, {rewards.shape=}, {target_return.shape=}, {timesteps.shape=}")
        model.max_length = None # HISTORY_LEN
        sampled_action = model.get_action(
                                            states=states_norm,
                                            actions=actions.to(dtype=torch.float32),
                                            rewards=rewards.to(dtype=torch.float32),
                                            returns_to_go=target_return.unsqueeze(-1),
                                            timesteps=timesteps.to(dtype=torch.long),
                                        )
            
        if not use_argmax:
            act = np.random.choice([0, 1, 2, 3, 4], p=torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())
        else:
            act = np.argmax(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy())        

        actions[-1] = act
        act_list.append(act)
        
        state, reward, done, info = env.step(act)
        state = np.float32(state['image'])
        state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])
        
        out_states.append(state)
        
        rews.append(reward)
        cur_state = torch.from_numpy(state).to(device=device).float()
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
    
    env.close()
    return episode_return, act_list, t, out_states, memories, None

# * ##############################################

# episode_timeout = 2100
# use_argmax = False
# episode_return, act_list1, t, states1, memories = get_returns_VizDoom(model=model, ret=56.5, seed=2, 
#                                                                       episode_timeout=episode_timeout, 
#                                                                       context_length=config["training_config"]["context_length"], 
#                                                                       device=device, 
#                                                                       config,
#                                                                       act_dim=config["model_config"]["ACTION_DIM"], 
#                                                                       use_argmax=use_argmax, create_video=True)

# print("Episode return:", episode_return)