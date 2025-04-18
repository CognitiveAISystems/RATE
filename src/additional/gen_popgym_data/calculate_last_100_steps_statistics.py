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
from typing import Optional
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
    env_idx: Optional[int] = None  # Environment index (1-48) to collect dataset
    number_of_episodes: int = 100 # 3000
    group: Optional[int] = None  # Group number (1-4) to collect datasets. If -1, collect all datasets sequentially

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

    total_rewards = []
    total_lengths = []

    for i in tqdm(range(number_of_episodes)):
        obsList, actList, rewList, doneList = [], [], [], []
        
        # Run inference
        episode_reward = 0
        obs, info = env.reset()
        done = False
        episode_length = 0  # Track the length of the episode

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
            episode_length += 1  # Increment episode length
            rewList.append(reward)
            actList.append(action)
            doneList.append(int(done))

        # Collect statistics
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)

        env.close()

    # Calculate averages over the last 100 episodes
    avg_length = np.mean(total_lengths[-100:]) if len(total_lengths) >= 100 else np.mean(total_lengths)
    avg_min_reward = np.min(total_rewards[-100:]) if len(total_rewards) >= 100 else np.min(total_rewards)
    avg_mean_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
    avg_max_reward = np.max(total_rewards[-100:]) if len(total_rewards) >= 100 else np.max(total_rewards)

    print(f"Average Episode Length: {avg_length}")
    print(f"Average Minimum Reward: {avg_min_reward}")
    print(f"Average Mean Reward: {avg_mean_reward}")
    print(f"Average Maximum Reward: {avg_max_reward}")

    # Write statistics to CSV file
    import csv

    # Check if the file exists to write headers only once
    file_exists = os.path.isfile('100_steps_stats.csv')

    with open('100_steps_stats.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is new
        if not file_exists:
            writer.writerow(['Environment Name', 'Average Length', 'Average Minimum Reward', 'Average Mean Reward', 'Average Maximum Reward'])
        writer.writerow([env_name, avg_length, avg_min_reward, avg_mean_reward, avg_max_reward])
        

    agent.stop()
    ray.shutdown()


def main(args: CLIArgs):

    assert (args.group is not None and args.env_idx is None) or (args.group is None and args.env_idx is not None)

    # 1) Download checkpoints
    download_ppo_checkpoints()

    # 2) Rename directories
    rename_ppo_dirs()

    if args.group is not None:
        print('\nDataset will be collected by groups, not by environment index\n')
        group_1 = np.arange(0, 12)
        group_2 = np.arange(12, 24)
        group_3 = np.arange(24, 36)
        group_4 = np.arange(36, 48)
        
        groups = {
            1: group_1,
            2: group_2,
            3: group_3,
            4: group_4
        }

        if args.group not in groups and args.group != -1:
            raise ValueError(f"Group must be between 1 and 4 (or -1 to collect all datasets sequentially), got {args.group}")
        
        # 3) Collect dataset
        print(f"Collecting datasets for group {args.group}")

        if args.group != -1:
            for env_idx in groups[args.group]:
                collect_data_by_env_idx(env_idx, number_of_episodes=args.number_of_episodes)
        else:
            for group_idx in groups:
                print(f"Collecting datasets for group {group_idx}")
                for env_idx in groups[group_idx]:
                    collect_data_by_env_idx(env_idx, number_of_episodes=args.number_of_episodes)

    else:
        print(f"Collecting datasets for environment index {args.env_idx}")

        # 3) Collect dataset
        collect_data_by_env_idx(args.env_idx, number_of_episodes=args.number_of_episodes)


# src/additional/gen_popgym_data/results/PPO
if __name__ == "__main__":
    # python3 src/additional/gen_popgym_data/gen_popgym_data.py --args.env-idx=1 --args.number-of-episodes=3000
    tyro.cli(main)
