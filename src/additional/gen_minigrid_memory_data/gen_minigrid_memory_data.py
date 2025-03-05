import numpy as np
import pickle
import torch
import os
from tqdm import tqdm

from model import ActorCriticModel
from utils import create_env

def init_transformer_memory(trxl_conf, max_episode_steps, device):
    """Returns initial tensors for the episodic memory of the transformer.

    Arguments:
        trxl_conf {dict} -- Transformer configuration dictionary
        max_episode_steps {int} -- Maximum number of steps per episode
        device {torch.device} -- Target device for the tensors

    Returns:
        memory {torch.Tensor}, memory_mask {torch.Tensor}, memory_indices {torch.Tensor} -- Initial episodic memory, episodic memory mask, and sliding memory window indices
    """
    # Episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((trxl_conf["memory_length"], trxl_conf["memory_length"])), diagonal=-1)
    # Episdic memory tensor
    memory = torch.zeros((1, max_episode_steps, trxl_conf["num_blocks"], trxl_conf["embed_dim"])).to(device)
    # Setup sliding memory window indices
    repetitions = torch.repeat_interleave(torch.arange(0, trxl_conf["memory_length"]).unsqueeze(0), trxl_conf["memory_length"] - 1, dim = 0).long()
    memory_indices = torch.stack([torch.arange(i, i + trxl_conf["memory_length"]) for i in range(max_episode_steps - trxl_conf["memory_length"] + 1)]).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    return memory, memory_mask, memory_indices


if __name__ == "__main__":
    model_path = 'src/additional/gen_minigrid_memory_data/MiniGrid-MemoryS13Random-v0.nn'
    print("MODEL PATH:", model_path)

    # Set inference device and default tensor type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))
    print(config)

    # Instantiate environment
    env_name =  {'type': 'Minigrid', 'name': 'MiniGrid-MemoryS13Random-v0'}
    env = create_env(env_name, length=11, render=True)

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,), env.max_episode_steps)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    i = 0
    NUMBER_OF_TRAIN_DATA = 10000

    while i < NUMBER_OF_TRAIN_DATA:
        obsList, actList, rewList, doneList = [], [], [], []
        # Run and render episode
        frames = []

        done = False
        episode_rewards = []
        memory, memory_mask, memory_indices = init_transformer_memory(config["transformer"], env.max_episode_steps, device)
        memory_length = config["transformer"]["memory_length"]
        t = 0
        env_name = config["environment"]
        env = create_env(env_name, length=11, render=True)
        obs = env.reset()
        while not done:
            # Prepare observation and memory
            obsList.append(obs)
            obs = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)
            in_memory = memory[0, memory_indices[t].unsqueeze(0)]
            t_ = max(0, min(t, memory_length - 1))
            mask = memory_mask[t_].unsqueeze(0)
            indices = memory_indices[t].unsqueeze(0)
        
            policy, value, new_memory = model(obs, in_memory, mask, indices)
            memory[:, t] = new_memory
            # Sample action
            action = []
            for action_branch in policy:
                action.append(action_branch.sample().item())
            # Step environemnt
            obs, reward, done, info = env.step(action)
            rewList.append(reward)
            actList.append(action[0])
            doneList.append(int(done))

            episode_rewards.append(reward)
            t += 1

        if np.mean(rewList) > 0 and len(rewList) >= 10 and len(rewList) <= 30:
            DATA = {'obs': np.array(obsList), # (10, 3, 84, 84)
                    'action': np.array(actList),
                    'reward': np.array(rewList),
                    'done': np.array(doneList)}
            file_path = f'data/Minigrid_Memory/'
            isExist = os.path.exists(file_path)
            if not isExist:
                os.makedirs(file_path)
                
            np.savez(file_path + f'train_data_{i}.npz', **DATA)
            i += 1

        print(f'[{i}], R {info["reward"]}, L {info["length"]} | collected [{i} / {NUMBER_OF_TRAIN_DATA}]', end='\r')

        env.close()

    """
    data = np.load('MinigridMemory_data/train_data_0.npz')
    s = data['obs'] * 255.0  # convert to 0-255 range
    a = data['action']
    r = data['reward']
    d = data['done']

    print(s.shape, a.shape, r.shape, d.shape)
    """