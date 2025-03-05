import os
import torch
import numpy as np
from tqdm import tqdm

import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.envs.vizdoom_two_colors import env_vizdoom2, doom_environment2, models2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

from tqdm import tqdm

import matplotlib.pyplot as plt
from itertools import count


env_args = {
    'simulator':'doom', 
    'scenario':'custom_scenario{:003}.cfg', #custom_scenario_no_pil{:003}.cfg
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
    'model_checkpoint':'src/additional/gen_vizdoom_data/two_col_p1_checkpoint_0198658048.pth.tar',
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = doom_environment2.DoomEnvironment(env_args, idx=0, is_train=True, get_extra_info=False)
    print("Number of env actions:", env.num_actions)
    obs_shape = (3, env_args['screen_height'], env_args['screen_width'])
    print(f"obs_shape: {obs_shape}")

    scene = 0
    scenario = env_args['scenario_dir'] + env_args['scenario'].format(scene) # 0 % 63
    config = scenario

    env = env_vizdoom2.DoomEnvironmentDisappear(
        scenario=config,
        show_window=False,
        use_info=True,
        use_shaping=False, #if False bonus reward if #shaping reward is always: +1,-1 in two_towers
        frame_skip=2,
        no_backward_movement=True)

    policy = models2.CNNPolicy((3, 64, 112), env_args).to(device)
    checkpoint = torch.load(env_args['model_checkpoint'], map_location=lambda storage, loc: storage) 
    policy.load_state_dict(checkpoint['model'])
    policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = doom_environment2.DoomEnvironment(env_args, idx=0, is_train=True, get_extra_info=False)
    obs_shape = (3, env_args['screen_height'], env_args['screen_width'])

    scene = 0
    scenario = env_args['scenario_dir'] + env_args['scenario'].format(scene) # 0 % 63
    config = scenario

    env = env_vizdoom2.DoomEnvironmentDisappear(
        scenario=config,
        show_window=False,
        use_info=True,
        use_shaping=False, #if False bonus reward if #shaping reward is always: +1,-1 in two_towers
        frame_skip=2,
        no_backward_movement=True)

    policy = models2.CNNPolicy((3, 64, 112), env_args).to(device)
    checkpoint = torch.load(env_args['model_checkpoint'], map_location=lambda storage, loc: storage) 
    policy.load_state_dict(checkpoint['model'])
    policy.eval()

    NUMBER_OF_TRAIN_DATA = 5000 # 5000
    EPISODE_TIMEOUT = 90 # 90

    returns_red, returns_green = [], []

    save_dir = 'data/ViZDoom_Two_Colors'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating {NUMBER_OF_TRAIN_DATA} episodes")
    for i in tqdm(range(NUMBER_OF_TRAIN_DATA)):
        obsList, actList, rewList, doneList, isRedList = [], [], [], [], []
        times = []
        obs = env.reset()
        state = torch.zeros(1, env_args['hidden_size']).to(device)
        mask = torch.ones(1,1).to(device)
        done = False

        for t in count():
            times.append(t)
            obsList.append(obs['image'])
            result = policy(torch.from_numpy(obs['image']).unsqueeze(0).to(device), state, mask)
            action, state = result['actions'], result['states']
            
            
            obs, reward, done, info = env.step(action.item())


            is_red = info['is_red']
            rewList.append(reward)
            actList.append(action.item())
            doneList.append(int(done))
            isRedList.append(is_red)

            if done or t == EPISODE_TIMEOUT-1:

                if is_red == 1.0:
                    returns_red.append(np.sum(rewList))
                else:
                    returns_green.append(np.sum(rewList))

                break
        
        DATA = {'obs': np.array(obsList), # (1152, 3, 64, 112)
                'action': np.array(actList),
                'reward': np.array(rewList),
                'done': np.array(doneList),
                'is_red': np.array(isRedList)}

        file_path = f'{save_dir}/train_data_{i}.npz'
        np.savez(file_path, **DATA)
        

    env.close()


    """
    episode = np.load(f'../VizDoom_data/iterative_data/train_data_{4999}.npz')
    episode = {key: episode[key] for key in episode.keys()}
    """