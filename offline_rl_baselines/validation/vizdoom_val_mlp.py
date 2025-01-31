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
from mlp_agent_cql_bc import DecisionMLP
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


def load_model(seed, exp_name, loss_mode, stacked_input):
    agent = DecisionMLP(4, 1, 128, mode='doom')
    
    run_name = f'{exp_name}_mlp_{loss_mode}_{seed}_stacked_{stacked_input}'
    print(run_name)
    model_path = f'offline_rl_baselines/ckpt/vizdoom_ckpt_mlp/{loss_mode}/{seed}/{run_name}.ckpt'
    
    agent.load_state_dict(torch.load(model_path))
    
    agent.eval()
    agent.to(agent.device)
    
    return agent

sys.path.append('../../VizDoom/VizDoom_notebooks/')
from VizDoom.VizDoom_notebooks.doom_environment2 import DoomEnvironment
import env_vizdoom2

scene = 0
scenario = env_args['scenario_dir'] + env_args['scenario'].format(scene) # 0 % 63
config = scenario
device = 'cuda:0'

env = env_vizdoom2.DoomEnvironmentDisappear(
    scenario=config,
    show_window=False,
    use_info=True,
    use_shaping=False, #if False bonus reward if #shaping reward is always: +1,-1 in two_towers
    frame_skip=2,
    no_backward_movement=True)

def create_args():
    parser = argparse.ArgumentParser(description='RATE VizDoom trainer') 
    parser.add_argument('--loss-mode',           type=str, default='bc',   help='')

    return parser


if __name__ == '__main__':
    args = create_args().parse_args()
    exp_name = 'doom'
    stacked_input = False
    loss_mode = args.loss_mode
    totals, reds, greens = [], [], []

    for i in range(1, 6+1):
        agent = load_model(
            seed=i,
            exp_name=exp_name,
            stacked_input=stacked_input,
            loss_mode=loss_mode, 
        )

        _ = agent.eval()
        _ = agent.to(agent.device)
        PATH = f'../vizdoom_ckpt_mlp/{loss_mode}/{i}/{exp_name}_mlp_{loss_mode}_{i}_stacked_{stacked_input}.ckpt'
        # PATH = f'Doom_lstm_SAR_90_CQL'

        # weights = torch.load(f"{PATH}.ckpt", map_location="cpu")

        # agent.load_state_dict(weights, strict=True)
        # agent.train()
        # agent.to(agent.device)


        EPISODE_TIMEOUT = 4200 # 90

        NUMBER_OF_TRAIN_DATA = 100
        returns_red, returns_green, returns_total = [], [], []
        agent.eval()

        for i in tqdm(range(NUMBER_OF_TRAIN_DATA)):
            obsList, actList, rewList, doneList, isRedList = [], [], [], [], []
            times = []
            obs = env.reset()
            # plt.imshow(obs['image'].transpose(1,2,0))
            # plt.show()
            state = torch.zeros(1, env_args['hidden_size']).to(device)
            mask = torch.ones(1,1).to(device)
            done = False
            # agent.init_hidden(1)
            action = 0
            rtg = 56.5

            for t in count():
                times.append(t)
                obsList.append(obs['image'])
                #result = policy(torch.from_numpy(obs['image']).unsqueeze(0).to(device), state, mask)
                #action, state = result['actions'], result['states']

                states = torch.from_numpy(obs['image']).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    q_values = []
                    for possible_action in range(0, 5):  # 5 возможных действия
                        action_tensor = torch.tensor([[[possible_action]]], 
                                                dtype=torch.float32, 
                                                device=device).long()
                        rtg_tensor = torch.tensor([[[rtg]]], 
                                                dtype=torch.float32, 
                                                device=device)#.long()
                        # if loss_mode == 'cql':
                        #     update_lstm_hidden = possible_action==4
                        # else:
                        #     update_lstm_hidden = True
                            
                        action_preds, q1, q2, _ = agent.forward(
                            states = states / 255.,
                            actions = action_tensor,
                            returns_to_go = rtg_tensor,
                            # update_hidden = update_lstm_hidden,
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
                obs, reward, done, info = env.step(action)
                rtg -= reward

                is_red = info['is_red']
                rewList.append(reward)
                actList.append(action)
                doneList.append(int(done))
                isRedList.append(is_red)

                if done or t == EPISODE_TIMEOUT-1:

                    if is_red == 1.0:
                        returns_red.append(np.sum(rewList))
                    else:
                        returns_green.append(np.sum(rewList))

                    returns_total.append(np.sum(rewList))

                    break

        print(f"\nResults for checkpoint {i}:")
        print(f"Red team average return:   {np.mean(returns_red):.2f}")
        print(f"Green team average return: {np.mean(returns_green):.2f}")
        print(f"Total average return:      {np.mean(returns_total):.2f}")
        print("-" * 50)

        totals.append(np.mean(returns_total))
        reds.append(np.mean(returns_red))
        greens.append(np.mean(returns_green))

    print('\n')
    print('#'*50)

    print(f'TOTAL: {np.mean(totals)} ± {sem(totals)}')
    print(f'RED: {np.mean(reds)} ± {sem(reds)}')
    print(f'GREEN: {np.mean(greens)} ± {sem(greens)}')