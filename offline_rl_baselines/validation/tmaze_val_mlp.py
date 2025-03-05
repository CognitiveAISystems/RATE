import argparse
import os
from tqdm import tqdm
import torch
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
# parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from mlp_agent_cql_bc import DecisionMLP
import yaml
import argparse
from tqdm import tqdm
from envs.tmaze.tmaze import TMazeClassicPassive
from src.utils.set_seed import seeds_list
import numpy as np
from tqdm import tqdm
import os
from scipy.stats import sem
import pandas as pd

def run_episode(seed, episode_timeout, corridor_length, stacked_input, loss_mode):
    channels = 5
    create_video = False

    env = TMazeClassicPassive(
        episode_length=episode_timeout, 
        corridor_length=corridor_length, 
        penalty=0, 
        seed=seed, 
        goal_reward=1.0)

    state = env.reset() # {x, y, hint}
    np.random.seed(seed)
    where_i = state[0]
    mem_state = state[2]
    mem_state2 = state

    state = np.concatenate((state, np.array([0]))) # {x, y, hint, flag}
    state = np.concatenate((state, np.array([np.random.randint(low=-1, high=1+1)]))) # {x, y, hint, flag, noise}

    if create_video == True:
        print("down, required act: 3" if mem_state == -1.0 else "up,  required act: 1")

    state = torch.tensor(state).reshape(1, 1, channels)
    out_states = []
    out_states.append(state.cpu().numpy())
    done = True
    Flag = 0
    rtg = 1.0
    # agent.init_hidden(1)
    
    episode_return, episode_length = 0, 0

    for t in range(episode_timeout):
        with torch.no_grad():
            q_values = []
            for possible_action in range(0, 3):  # 4 возможных действия
                action_tensor = torch.tensor([[[possible_action]]], 
                                        dtype=torch.float32, 
                                        device=agent.device).long()
                rtg_tensor = torch.tensor([[[rtg]]], 
                                        dtype=torch.float32, 
                                        device=agent.device)#.long()
                    
                action_preds, q1, q2, _ = agent.forward(
                    states = state[:, :, 1:].cuda().float(),
                    actions = action_tensor.cuda(),
                    returns_to_go = rtg_tensor.cuda(),
                    stacked_input = stacked_input,
                )

                q_value = torch.minimum(q1, q2)
                q_values.append(q_value)

                if loss_mode == 'bc':
                    break

            # Select action with max Q-value
            if loss_mode == 'cql':
                q_values = torch.cat(q_values, dim=-1)
                action = torch.argmax(q_values).item() #+ 3
            else:
                action = torch.argmax(torch.softmax(action_preds, dim=-1).squeeze()).item()

        # print(t, action)


        state, reward, done, info = env.step(action)

        rtg -= reward
        
        if t < 0:
            state[2] = mem_state2[2]
        
            # {x, y, hint} -> {x, y, hint, flag}
        if state[0] != env.corridor_length:
            state = np.concatenate((state, np.array([0])))
        else:
            if Flag != 1:
                state = np.concatenate((state, np.array([1])))
                Flag = 1
            else:
                state = np.concatenate((state, np.array([0])))

        state = np.concatenate((state, np.array([np.random.randint(low=-1, high=1+1)])))
        state = state.reshape(1, 1, channels)
        state = torch.from_numpy(state).float().cuda()
                
            
        if done:
            if create_video == True:
                if np.round(where_i, 4) == np.round(corridor_length, 4):
                    print("Junction achieved 😀 ✅✅✅")
                    print("Chosen act:", "up" if action == 1 else "down" if action == 3 else "wrong")
                    if mem_state == -1 and action == 3:
                        print("Correct choice 😀 ✅✅✅")
                    elif mem_state == 1 and action == 1:
                        print("Correct choice 😀 ✅✅✅")
                    else:
                        print("Wrong choice 😭 ⛔️⛔️⛔️")
                else:
                    print("Junction is not achieved 😭 ⛔️⛔️⛔️")
            break 

    return reward

def create_args():
    parser = argparse.ArgumentParser(description='RATE VizDoom trainer') 
    parser.add_argument('--loss-mode',           type=str, default='bc',   help='')
    return parser


if __name__ == '__main__':
    args = create_args().parse_args()
    exp_name = 'tmaze_mlp'
    stacked_input = False
    loss_mode = args.loss_mode
    output_dir = 'offline_rl_baselines/results'
    segments = 3

    # Setup output path
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f'{output_dir}/{exp_name}_{loss_mode}_{stacked_input}_{segments}_validation_results.csv'

    # Create empty DataFrame if file doesn't exist
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['exp_name', 'stacked_input', 'loss_mode', 'segments', 
                                'context_length', 'episode_timeout', 'corridor_length', 'mean', 'sem'])
        df.to_csv(csv_path, index=False)

    for context_length in tqdm(
        [3, 10, 30, 90],
        desc="Context lengths",
        position=0,
        leave=True
    ):
        for episode_timeout in tqdm(
            [9, 30, 60, 90, 150, 210, 270, 360, 480, 600, 750, 900],
            desc=f"Episode timeouts (context_len={context_length})",
            position=1,
            leave=False
        ):
            corridor_length = episode_timeout - 2

            means = []
            ckpt_dir = f'offline_rl_baselines/ckpt/tmaze_ckpt_mlp/{loss_mode}'
            for RUN in range(1, 11):
                ckpt = f'{ckpt_dir}/{RUN}/{exp_name}_{loss_mode}_{RUN}_stacked_{stacked_input}_context_{context_length}_segments_{segments}.ckpt'
                # print(ckpt)
                agent = DecisionMLP(4, 1, 32, num_layers=2, mode='tmaze')
                agent.load_state_dict(torch.load(ckpt))
                agent.eval()
                agent.to(agent.device)

                rewards = []
                for seed in tqdm(seeds_list, desc="Seeds", position=2, leave=False):
                    reward = run_episode(seed=seed, episode_timeout=episode_timeout, corridor_length=corridor_length, stacked_input=stacked_input, loss_mode=loss_mode)
                    rewards.append(reward)

                mean_, std_, = np.mean(rewards), np.std(rewards)
                means.append(mean_)

                # print('\n')

            total_mean, total_sem = np.mean(means), sem(means)
            
            # Create result dictionary
            result = {
                'exp_name': [exp_name],
                'stacked_input': [stacked_input],
                'loss_mode': [loss_mode],
                'segments': [segments],
                'context_length': [context_length],
                'episode_timeout': [episode_timeout],
                'corridor_length': [corridor_length],
                'mean': [total_mean],
                'sem': [total_sem]
            }
            
            # Convert to DataFrame and append to CSV
            df_new = pd.DataFrame(result)
            df_new.to_csv(csv_path, mode='a', header=False, index=False)
            
            # print("exp_name:", exp_name, "stacked_input:", stacked_input, "loss_mode:", loss_mode, 
            #     "segments:", segments, "context_length:", context_length, "episode_timeout:", episode_timeout, "corridor_length:", corridor_length,
            #     "total_mean:", total_mean, "total_sem:", total_sem)
            # print(f"Results for context_length {context_length} saved to: {csv_path}")