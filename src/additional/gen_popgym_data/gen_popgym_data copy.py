import wget
import zipfile
import os

import ray
import torch
from ray.tune.registry import register_env
import popgym
from popgym import wrappers
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.core.env import POPGymEnv
import numpy as np
from dataclasses import dataclass
import tyro
from ray.rllib.algorithms.ppo import PPO
import os
from tqdm import tqdm

from get_dataset_collectors_ckpt import env_names_dict

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""


@dataclass
class CLIArgs:
    """Options that will be passed through tyro."""
    env_idx: int = 39
    number_of_episodes: int = 3000

# python3 src/additional/gen_popgym_data/gen_popgym_data.py
# Define paths
path_to_data = 'data/POPGym/'
path_to_checkpoints = 'src/additional/gen_popgym_data/'

# Define target directory for extracted files
target_ppo_dir = os.path.join(path_to_checkpoints, 'results/PPO')

# Define zip file path
zip_file_path = os.path.join(path_to_checkpoints, 'PPO.zip')


def download_ppo_checkpoints():
    # Download and extract only if necessary
    if not os.path.exists(target_ppo_dir):
        # 1. Download agent's checkpoints from HuggingFace only if zip doesn't exist
        if not os.path.exists(zip_file_path):
            checkpoint_url = 'https://huggingface.co/datasets/avanturist/rate-popgym-checkpoints/resolve/main/PPO.zip'
            wget.download(checkpoint_url, zip_file_path)
        else:
            print("PPO checkpoints zip file already exists, skipping download")

        # 2. Unzip checkpoints
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(path_to_checkpoints)
        
        # 3. Remove zip file after extraction
        os.remove(zip_file_path)
        print(f"\nRemoved {zip_file_path}")
    else:
        print("PPO checkpoints directory already exists, skipping download")


def rename_ppo_dirs(base_dir="src/additional/gen_popgym_data/results/PPO", verbose=False):
    for dir_name in os.listdir(base_dir):
        old_path = os.path.join(base_dir, dir_name)

        # Check if it's a directory (not a file)
        if os.path.isdir(old_path) and dir_name.startswith("PPO_"):
            # Remove prefix "PPO_"
            new_name_part = dir_name.replace("PPO_", "", 1)
            # Split by first underscore and take everything before it
            # "popgym-AutoencodeEasy-v0_a1568..." -> "popgym-AutoencodeEasy-v0"
            new_dir_name = new_name_part.split("_", 1)[0]
            new_path = os.path.join(base_dir, new_dir_name)

            # If new folder doesn't exist, rename it
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                if verbose:
                    print(f"Renamed '{dir_name}' to '{new_dir_name}'")
            else:
                if verbose:
                    print(f"Skipped: '{new_dir_name}' already exists.")




def collect_data_by_env_idx(env_idx, number_of_episodes=3000):
    env_name = env_names_dict[env_idx][0]
    print(env_name)
    env_class = env_names_dict[env_idx][1]


    def wrap(env: POPGymEnv) -> POPGymEnv:
        return wrappers.Antialias(wrappers.PreviousAction(env))

    env_name = env_names_dict[env_idx][0]
    env_class = env_names_dict[env_idx][1]
    register_env(env_name, lambda x: wrap(env_class()))

    env = env_class()  # Create base environment

    # Apply wrappers because the PPO-GRU agent required to collect data
    # was trained in online mode with these wrappers
    # When collecting data, we turned off this wrappers because we
    # solving supervised learning task
    env = wrap(env)    

    # Configuration parameters
    bptt_cutoff = 1024
    num_workers = 4
    num_envs_per_worker = 16
    gpu_per_worker = 0.25
    max_steps = 15_000_000

    # Hidden sizes
    h = 128  # Hidden size of linear layers
    h_memory = 256  # Hidden size of memory

    config = {
        "env": env_name,
        "framework": "torch",
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "num_gpus": gpu_per_worker,
        
        # PPO specific parameters from the table
        "gamma": 0.99,                    # Decay factor
        "vf_loss_coeff": 1.0,            # Value function loss coefficient
        "entropy_coeff": 0.0,            # Entropy loss coefficient
        "lr": 5e-5,                      # Learning rate
        "num_sgd_iter": 30,              # Number of SGD iterations
        "train_batch_size": 65536,       # Batch size
        "sgd_minibatch_size": 8192,      # Minibatch size
        "clip_param": 0.3,               # PPO clipping parameter
        "vf_clip_param": 0.3,            # Value function clipping
        "kl_target": 0.01,               # KL target
        "kl_coeff": 0.2,                 # KL coefficient
        
        # Other parameters
        "horizon": 1024,                 # Maximum Episode Length
        "batch_mode": "complete_episodes",
        "rollout_fragment_length": bptt_cutoff,
        
        "model": {
            "max_seq_len": bptt_cutoff,
            "custom_model": GRU,
            "custom_model_config": {
                "preprocessor_input_size": h,
                "preprocessor": torch.nn.Sequential(
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "preprocessor_output_size": h,
                "hidden_size": h_memory,
                "postprocessor": torch.nn.Identity(),
                "actor": torch.nn.Sequential(
                    torch.nn.Linear(h_memory, h),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "critic": torch.nn.Sequential(
                    torch.nn.Linear(h_memory, h),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "postprocessor_output_size": h,
            },
        },
    }

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    checkpoint_path = f"{target_ppo_dir}/{env_name}/checkpoint_000220"
    agent = PPO(config=config)
    agent.restore(checkpoint_path)

    for i in tqdm(range(number_of_episodes)):
        obsList, actList, rewList, doneList = [], [], [], []
        
        # Run inference
        episode_reward = 0
        obs, info = env.reset()
        # print(obs, info)
        done = False

        # Initialize RNN state
        state = agent.get_policy().get_initial_state()

        while not done:
            obsList.append(obs[:-2])

            # Get action from agent considering RNN state
            action, state_out, _ = agent.compute_single_action(
                observation=obs,
                state=state,
                explore=False  # Turn off exploration for deterministic behavior
            )
            
            # Update RNNs state
            state = state_out
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            rewList.append(reward)
            actList.append(action)
            doneList.append(int(done))
            # print(obs, action, reward, done, info, episode_reward)
            # env.render()

        # print(f"Total episode reward: {episode_reward}")
        DATA = {
            'obs': np.array(obsList),
            'action': np.array(actList),
            'reward': np.array(rewList),
            'done': np.array(doneList)
        }

        file_path = f'data/POPGym/{env_name}/'
        os.makedirs(file_path, exist_ok=True)
        np.savez(file_path + f'train_data_{i}.npz', **DATA)
        env.close()

    agent.stop()
    ray.shutdown()


def main(args: CLIArgs):
    # 1) Скачиваем чекпоинты
    download_ppo_checkpoints()

    # 2) Переименовываем директории
    rename_ppo_dirs()

    # 3) Собираем датасет
    collect_data_by_env_idx(args.env_idx, number_of_episodes=args.number_of_episodes)


# src/additional/gen_popgym_data/results/PPO
if __name__ == "__main__":
    # python3 src/additional/gen_popgym_data/gen_popgym_data.py --args.env-idx=1 --args.number-of-episodes=3000

    tyro.cli(main)
