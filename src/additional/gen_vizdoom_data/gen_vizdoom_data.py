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

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import math


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

# Number of steps in episode
EPISODE_TIMEOUT = 150

def collect_episodes(rank, num_episodes, save_dir, env_args, device_id, queue):
    # Set device for this process
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    # Initialize environment
    env = doom_environment2.DoomEnvironment(env_args, idx=0, is_train=True, get_extra_info=False)
    scene = 0
    scenario = env_args['scenario_dir'] + env_args['scenario'].format(scene)
    config = scenario

    env = env_vizdoom2.DoomEnvironmentDisappear(
        scenario=config,
        show_window=False,
        use_info=True,
        use_shaping=False,
        frame_skip=2,
        no_backward_movement=True
    )

    # Initialize policy
    policy = models2.CNNPolicy((3, 64, 112), env_args).to(device)
    checkpoint = torch.load(env_args['model_checkpoint'], map_location=lambda storage, loc: storage)
    policy.load_state_dict(checkpoint['model'])
    policy.eval()

    returns_red, returns_green = [], []
    
    for i in range(num_episodes):
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
        
        DATA = {
            'obs': np.array(obsList),
            'action': np.array(actList),
            'reward': np.array(rewList),
            'done': np.array(doneList),
            'is_red': np.array(isRedList)
        }

        episode_idx = rank * num_episodes + i
        file_path = f'{save_dir}/train_data_{episode_idx}.npz'
        np.savez(file_path, **DATA)
        queue.put(1)

    env.close()

if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA multiprocessing
    mp.set_start_method('spawn')
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available. This script requires at least one GPU.")
    
    # Configuration
    NUMBER_OF_TRAIN_DATA = 5000
    NUM_PROCESSES = min(num_gpus * 8, 8)  # Use 2 processes per GPU, but no more than 8
    EPISODES_PER_PROCESS = math.ceil(NUMBER_OF_TRAIN_DATA / NUM_PROCESSES)
    
    # Create save directory
    save_dir = f'data/ViZDoom_Two_Colors_{EPISODE_TIMEOUT}'
    os.makedirs(save_dir, exist_ok=True)

    # Progress tracking
    progress_queue = Queue()
    
    print(f"Starting data collection with {NUM_PROCESSES} processes")
    print(f"Episodes per process: {EPISODES_PER_PROCESS}")
    
    # Start processes
    processes = []
    for rank in range(NUM_PROCESSES):
        device_id = rank % num_gpus
        p = Process(target=collect_episodes, 
                   args=(rank, EPISODES_PER_PROCESS, save_dir, env_args, device_id, progress_queue))
        p.start()
        processes.append(p)
    
    # Monitor progress
    total_episodes = 0
    pbar = tqdm(total=NUMBER_OF_TRAIN_DATA)
    while total_episodes < NUMBER_OF_TRAIN_DATA:
        progress_queue.get()
        total_episodes += 1
        pbar.update(1)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    pbar.close()
    print("Data collection completed!")


    """
    episode = np.load(f'../VizDoom_data/iterative_data/train_data_{4999}.npz')
    episode = {key: episode[key] for key in episode.keys()}
    """