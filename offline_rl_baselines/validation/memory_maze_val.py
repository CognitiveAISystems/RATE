import torch
import sys
sys.path.append('../../../')
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
# parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from lstm_agent_cql_bc import DecisionLSTM
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import os

import matplotlib.pyplot as plt

import torch
import numpy as np

sys.path.append('VizDoom/VizDoom_notebooks/')
from VizDoom.VizDoom_notebooks.doom_environment2 import DoomEnvironment
import env_vizdoom2
from TMaze_new.TMaze_new_src.utils import set_seed

import pickle
from tqdm import tqdm
#import env_vizdoom2
import matplotlib.pyplot as plt
from itertools import count
import time
import random
from scipy.stats import sem

import torch
import numpy as np
import gym
gym.logger.set_level(40)
import logging
logging.disable(logging.WARNING)
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
import warnings
warnings.filterwarnings('ignore')



def load_model(seed, exp_name, loss_mode, stacked_input):
    agent = DecisionLSTM(3, 6, 128, mode='memory_maze')
    
    run_name = f'{exp_name}_{loss_mode}_{seed}_stacked_{stacked_input}'
    print(run_name)
    model_path = f'offline_rl_baselines/ckpt/memory_maze_ckpt/{loss_mode}/{seed}/{run_name}.ckpt'
    
    agent.load_state_dict(torch.load(model_path))
    
    agent.eval()
    agent.to(agent.device)
    
    return agent

sys.path.append('../../VizDoom/VizDoom_notebooks/')
from VizDoom.VizDoom_notebooks.doom_environment2 import DoomEnvironment
import env_vizdoom2


device = 'cuda:0'


def create_args():
    parser = argparse.ArgumentParser(description='RATE VizDoom trainer') 
    parser.add_argument('--loss-mode',           type=str, default='bc',   help='')
    parser.add_argument('--seed',           type=int, default='1',   help='')

    return parser

# python offline_rl_baselines/validation/memory_maze_val.py --loss-mode=bc --seed=1
if __name__ == '__main__':
    args = create_args().parse_args()
    exp_name = 'memory_maze'
    stacked_input = False
    loss_mode = args.loss_mode
    totals, reds, greens = [], [], []

    # for i in range(1, 3+1):
    for i in range(args.seed, 3+1):
        agent = load_model(
            seed=i,
            exp_name=exp_name,
            stacked_input=stacked_input,
            loss_mode=loss_mode, 
        )

        _ = agent.eval()
        _ = agent.to(agent.device)
        PATH = f'../memory_maze_ckpt/{loss_mode}/{i}/{exp_name}_{loss_mode}_{i}_stacked_{stacked_input}.ckpt'
        # PATH = f'Doom_lstm_SAR_90_CQL'

        # weights = torch.load(f"{PATH}.ckpt", map_location="cpu")

        # agent.load_state_dict(weights, strict=True)
        # agent.train()
        # agent.to(agent.device)


        EPISODE_TIMEOUT = 1000 # 90

        NUMBER_OF_TRAIN_DATA = 100
        returns_red, returns_green, returns_total = [], [], []
        agent.eval()

        for j in tqdm(range(NUMBER_OF_TRAIN_DATA)):
            obsList, actList, rewList, doneList, isRedList = [], [], [], [], []
            env = gym.make('memory_maze:MemoryMaze-9x9-v0', seed=j)
            times = []
            state = env.reset()
            # plt.imshow(obs['image'].transpose(1,2,0))
            # plt.show()
            state = state.transpose(2, 0, 1) # model trained on [C, H, W], but env returns [H, W, C]
            # state = state.reshape(1, 1, state.shape[0], state.shape[1], state.shape[2])

            mask = torch.ones(1,1).to(device)
            done = False
            agent.init_hidden(1)
            action = 0
            rtg = 18.1

            for t in count():
                times.append(t)
                #result = policy(torch.from_numpy(obs['image']).unsqueeze(0).to(device), state, mask)
                #action, state = result['actions'], result['states']

                states = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device).float()

                with torch.no_grad():
                    q_values = []
                    for possible_action in range(0, 6): 
                        action_tensor = torch.tensor([[[possible_action]]], 
                                                dtype=torch.float32, 
                                                device=device).long()
                        rtg_tensor = torch.tensor([[[rtg]]], 
                                                dtype=torch.float32, 
                                                device=device)#.long()
                        if loss_mode == 'cql':
                            update_lstm_hidden = possible_action==5
                        else:
                            update_lstm_hidden = True
                        
                        action_preds, q1, q2, _ = agent.forward(
                            states = states.float() / 255.,
                            actions = action_tensor,
                            returns_to_go = rtg_tensor,
                            update_hidden = update_lstm_hidden,
                            stacked_input = stacked_input,
                        )
                        q_value = torch.minimum(q1, q2)
                        q_values.append(q_value)

                        if not loss_mode == 'cql':
                            break

                    # Select action with max Q-value
                    if loss_mode == 'cql':
                        q_values = torch.cat(q_values, dim=-1)
                        action = torch.argmax(q_values).item() #+ 3
                    else:
                        action = torch.argmax(torch.softmax(action_preds, dim=-1).squeeze()).item()

                #action = random.choice([3,4])
                #print(t,action, q_values)
                # print(action)
                state, reward, done, info = env.step(action)
                # plt.imshow(state)
                # plt.show()
                state = state.transpose(2, 0, 1)
                rtg -= reward

                rewList.append(reward)
                actList.append(action)
                doneList.append(int(done))
                

                if done or t == EPISODE_TIMEOUT-1:
                    returns_total.append(np.sum(rewList))
                    if j < 5:
                        print(np.sum(rewList))
                    break

        print(f"\nResults for checkpoint {i}:")
        print(f"Total average return:      {np.mean(returns_total):.2f}")
        print("-" * 50)

        totals.append(np.mean(returns_total))

    print('\n')
    print('#'*50)

    print(f'TOTAL: {np.mean(totals)} ± {sem(totals)}')