"""
ARShot Dataset Class

A PyTorch Dataset for loading and processing ARShot environment trajectories.
Compatible with the existing RATE/ELMUR training pipeline.
Follows the same structure as TMaze dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import pickle
import glob
from typing import Tuple, Optional, Dict, Any, List
import torch.nn.functional as F
from tqdm import tqdm
from src.envs.associative_retrieval.arshotenv import ARShotEnv


def ARShot_data_generator(n_pairs: int, shot_mode: str, num_episodes: int, 
                         max_length: int = 100, **env_kwargs) -> None:
    """Generate and save ARShot trajectory data.

    This function generates trajectory data for ARShot environments with specified
    configuration and saves them as pickle files. It ensures data availability for training
    by generating missing data files.

    Args:
        n_pairs: Number of key-value pairs in the environment.
        shot_mode: Mode for shot placement ("after_pairs" or "after_any_colon").
        num_episodes: Number of episodes to generate.
        max_length: Maximum sequence length for trajectory segments.
        **env_kwargs: Additional keyword arguments for ARShotEnv.

    Note:
        Generated data is saved in 'data/ARShot/' directory with filenames
        following the pattern 'arshot_n{n_pairs}_{shot_mode}_ep{num_episodes}.pickle'.
    """
    current_directory = os.getcwd()
    name = f'arshot_n{n_pairs}_{shot_mode}_ep{num_episodes}'
    data_path = os.path.join(current_directory, 'data', 'ARShot')
    save_path = os.path.join(data_path, f"{name}.pickle")
    
    if not os.path.exists(save_path):
        print("ARShot data is not available. Generating...")
        generate_dict_with_arshot_trajectories(n_pairs=n_pairs, shot_mode=shot_mode,
                                             num_episodes=num_episodes, max_length=max_length,
                                             **env_kwargs)
        print("ARShot data successfully generated.")
    else:
        print("ARShot data is available.")


def generate_arshot_trajectory(n_pairs: int, shot_mode: str, episode_seed: int,
                              max_length: int = 100, **env_kwargs) -> tuple:
    """Generate a single ARShot trajectory.

    This function generates a trajectory in the ARShot environment following
    the correct policy for associative retrieval.

    Args:
        n_pairs: Number of key-value pairs in the environment.
        shot_mode: Mode for shot placement ("after_pairs" or "after_any_colon").
        episode_seed: Seed for reproducible episode generation.
        max_length: Maximum sequence length.
        **env_kwargs: Additional keyword arguments for ARShotEnv.

    Returns:
        A tuple containing:
            - observations (list): List of token strings
            - actions (list): List of action token strings  
            - rtgs (ndarray): Return-to-go values
            - dones (ndarray): Episode termination flags
            - info (dict): Episode info containing mapping and query_key
    """
    # Create environment with seed
    env = ARShotEnv(n_pairs=n_pairs, shot_mode=shot_mode, rng_seed=episode_seed, **env_kwargs)
    
    obs, info = env.reset()
    done = False
    t = 0
    
    observations = []
    actions = []
    rewards = []
    dones = []
    
    # Add initial observation
    obs_token = env.id_to_token[obs]
    observations.append(obs_token)
    
    while not done and t < max_length:
        # Determine action based on current token
        current_token = env.id_to_token[obs]
        
        if current_token == "shot":
            # When we see 'shot', we should predict the correct value
            act_token = info["mapping"][info["query_key"]]
        else:
            # For all other tokens, use 'pass' (no-op)
            act_token = "pass"
        
        # Take action
        action_id = env.token_to_id[act_token]
        next_obs, reward, done, truncated, next_info = env.step(action_id)
        
        actions.append(act_token)
        rewards.append(reward)
        dones.append(done or truncated)
        
        # Add next observation if not done
        if not (done or truncated) and t + 1 < max_length:
            next_obs_token = env.id_to_token[next_obs]
            observations.append(next_obs_token)
        
        obs = next_obs
        t += 1
    
    # Convert rewards to return-to-go
    rewards = np.array(rewards)
    rtgs = np.zeros_like(rewards, dtype=np.float32)
    rtg = 0.0
    for i in reversed(range(len(rewards))):
        rtg = rewards[i] + rtg  # gamma = 1.0 for ARShot
        rtgs[i] = rtg
    
    # Add initial RTG
    rtgs = np.concatenate([[rtgs[0]], rtgs])
    
    return observations, actions, rtgs, np.array(dones), info


def generate_dict_with_arshot_trajectories(n_pairs: int, shot_mode: str, num_episodes: int,
                                          max_length: int = 100, **env_kwargs) -> None:
    """Generate and save a dictionary of ARShot trajectories.

    This function generates multiple trajectories for the ARShot environment
    and saves them as a dictionary in a pickle file.

    Args:
        n_pairs: Number of key-value pairs in the environment.
        shot_mode: Mode for shot placement ("after_pairs" or "after_any_colon").
        num_episodes: Number of episodes to generate.
        max_length: Maximum sequence length.
        **env_kwargs: Additional keyword arguments for ARShotEnv.

    Note:
        Generated data is saved as a dictionary where each key is an index
        and the value is a dictionary containing:
        - 'obs': List of observation tokens
        - 'action': List of action tokens
        - 'rtg': Return-to-go values
        - 'done': Episode termination flags
        - 'info': Episode info (mapping and query_key)
    """
    current_directory = os.getcwd()
    name = f'arshot_n{n_pairs}_{shot_mode}_ep{num_episodes}'
    data_path = os.path.join(current_directory, 'data', 'ARShot')
    save_path = os.path.join(data_path, f"{name}.pickle")
    
    # Create directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Remove existing files
    files = glob.glob(save_path + '*')
    for f in files:
        os.remove(f)

    data = {}
    
    for episode_idx in tqdm(range(num_episodes), desc="Generating ARShot episodes"):
        episode_seed = episode_idx + 1000  # Reproducible seeds
        
        observations, actions, rtgs, dones, info = generate_arshot_trajectory(
            n_pairs=n_pairs, shot_mode=shot_mode, episode_seed=episode_seed,
            max_length=max_length, **env_kwargs
        )
        
        data[episode_idx] = {
            'obs': observations,
            'action': actions, 
            'rtg': rtgs,
            'done': dones,
            'info': info
        }

    # Save data
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved {num_episodes} ARShot episodes to {save_path}")


class ARShotDataset(Dataset):
    """A PyTorch Dataset for loading and processing ARShot environment trajectories.
    
    This dataset handles loading and preprocessing of ARShot environment trajectories
    stored as pickle files. It processes observations (tokens), actions (tokens), and 
    rewards from the associative retrieval environment with 'shot' queries.

    Attributes:
        data (dict): Dictionary containing trajectory data loaded from pickle file.
        gamma (float): Discount factor for computing return-to-go (RTG).
        max_length (int): Maximum sequence length for trajectory segments.
        env (ARShotEnv): Reference environment for token-to-id mapping.

    Note:
        The dataset expects each trajectory to contain:
        - 'obs': List of observation tokens
        - 'action': List of action tokens
        - 'rtg': Return-to-go values
        - 'done': Episode termination flags
        - 'info': Episode info (mapping and query_key)
    """

    def __init__(self, path: str, gamma: float, max_length: int, n_pairs: int, 
                 shot_mode: str, **env_kwargs):
        """Initialize the ARShot dataset.

        Args:
            path: Path to the pickle file containing trajectory data.
            gamma: Discount factor for computing return-to-go values.
            max_length: Maximum sequence length for trajectory segments.
            n_pairs: Number of key-value pairs in the environment.
            shot_mode: Mode for shot placement ("after_pairs" or "after_any_colon").
            **env_kwargs: Additional keyword arguments for ARShotEnv.
        """
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.gamma = gamma
        self.max_length = max_length
        
        # Create reference environment for token-to-id mapping
        self.env = ARShotEnv(n_pairs=n_pairs, shot_mode=shot_mode, **env_kwargs)
        self.vocab_size = len(self.env.vocab)
        
    def discount_cumsum(self, x: np.ndarray) -> np.ndarray:
        """Compute the discounted cumulative sum of rewards.

        Args:
            x: 1D numpy array of reward values.

        Returns:
            A 1D numpy array containing the discounted cumulative sums for each timestep.
        """
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum

    def __getitem__(self, index: int) -> tuple:
        """Get a trajectory by index.

        Args:
            index: Index of the trajectory to retrieve.

        Returns:
            A tuple containing:
                - s (torch.Tensor): Observations as token indices of shape (max_length,)
                - a (torch.Tensor): Actions as token indices of shape (max_length,)
                - rtg (torch.Tensor): Return-to-go values of shape (max_length, 1)
                - d (torch.Tensor): Done flags of shape (max_length,)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length,)
                - mask (torch.Tensor): Mask tensor of shape (max_length,)

        Note:
            For SA sequence format, targets are generated as:
            - Next token in sequence for most positions
            - Correct answer for positions where observation is 'shot'
            Shorter trajectories are padded to max_length with -10 for actions.
        """
        traj = self.data[index]
        
        # Get trajectory data
        obs_tokens = traj['obs']
        action_tokens = traj['action']
        rtgs = traj['rtg']
        dones = traj['done']
        info = traj['info']
        
        # Convert tokens to indices
        obs_indices = []
        for token in obs_tokens:
            obs_indices.append(self.env.token_to_id.get(token, self.env.token_to_id.get('pass', 0)))
        
        # For SA format, targets should be the ACTIONS, not next observations
        # Model learns: when obs="shot" → predict correct_value, otherwise → predict "pass"
        target_indices = []
        for i in range(len(obs_tokens)):
            if i < len(action_tokens):
                # Use the actual action from the episode
                action_token = action_tokens[i]
                if action_token == -10:
                    target_indices.append(-10)  # Padding
                else:
                    target_indices.append(self.env.token_to_id.get(action_token, self.env.token_to_id.get('pass', 0)))
            else:
                # Padding beyond action sequence
                target_indices.append(-10)
        
        length = len(obs_indices)
        
        # Convert to tensors
        s = torch.tensor(obs_indices, dtype=torch.long)
        a = torch.tensor(target_indices, dtype=torch.long)  # Contains next-token targets
        rtg = torch.from_numpy(rtgs).float()
        d = torch.from_numpy(dones).long()
        timesteps = torch.arange(length, dtype=torch.long)
        mask = torch.ones(length, dtype=torch.float)
        
        # Ensure RTG and dones have the right length (same as observations)
        if len(rtg) != length:
            # Pad or truncate RTG to match observation length
            if len(rtg) < length:
                rtg = torch.cat([rtg, torch.zeros(length - len(rtg))])
            else:
                rtg = rtg[:length]
        
        if len(d) != length:
            # Pad or truncate dones to match observation length
            if len(d) < length:
                d = torch.cat([d, torch.zeros(length - len(d), dtype=torch.long)])
            else:
                d = d[:length]
        
        # Reshape to match expected format
        s = s.reshape(1, length, 1)
        a = a.reshape(1, length, 1)
        rtg = rtg.reshape(1, length, 1)
        d = d.reshape(1, length, 1)
        timesteps = timesteps.reshape(1, length, 1)
        mask = mask.reshape(1, length, 1)
        
        # Pad if necessary
        if length < self.max_length:
            pad_length = self.max_length - length
            
            # Pad observations with last observation
            last_obs = s[:, -1:, :]
            s = torch.cat([s, last_obs.repeat(1, pad_length, 1)], dim=1)
            
            # Pad targets with -10
            a = F.pad(a, (0, 0, 0, pad_length, 0, 0), value=-10)
            
            # Pad other tensors
            d = F.pad(d, (0, 0, 0, pad_length, 0, 0), value=2)
            timesteps = F.pad(timesteps, (0, 0, 0, pad_length, 0, 0), value=0)
            mask = F.pad(mask, (0, 0, 0, pad_length, 0, 0), value=0)
            rtg = F.pad(rtg, (0, 0, 0, pad_length, 0, 0), value=0)
        
        return s.squeeze(0), a.squeeze(0), rtg.squeeze(0), d.squeeze(0), timesteps.squeeze(0), mask.squeeze(0)
    
    def __len__(self):
        return len(self.data)


def create_arshot_dataloader(n_pairs: int = 6, shot_mode: str = "after_pairs",
                            max_length: int = 100, gamma: float = 1.0, 
                            num_episodes: int = 1000, batch_size: int = 32,
                            shuffle: bool = True, num_workers: int = 0,
                            **env_kwargs) -> DataLoader:
    """Create a DataLoader for ARShot datasets following TMaze pattern.
    
    Args:
        n_pairs: Number of key-value pairs in the environment.
        shot_mode: Mode for shot placement ("after_pairs" or "after_any_colon").
        max_length: Maximum sequence length.
        gamma: Discount factor for return-to-go computation.
        num_episodes: Number of episodes to generate.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        **env_kwargs: Additional environment kwargs.
        
    Returns:
        A PyTorch DataLoader for the ARShot dataset.
    """
    # Ensure data is generated
    ARShot_data_generator(n_pairs=n_pairs, shot_mode=shot_mode, 
                         num_episodes=num_episodes, max_length=max_length, 
                         **env_kwargs)
    
    # Create dataset path
    name = f'arshot_n{n_pairs}_{shot_mode}_ep{num_episodes}'
    data_path = os.path.join(os.getcwd(), 'data', 'ARShot', f"{name}.pickle")
    
    # Create dataset
    dataset = ARShotDataset(
        path=data_path,
        gamma=gamma,
        max_length=max_length,
        n_pairs=n_pairs,
        shot_mode=shot_mode,
        **env_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    # Test the restructured dataset
    print("Testing restructured ARShot dataset...")
    
    # Test data generation and loading
    dataloader = create_arshot_dataloader(
        n_pairs=3,
        shot_mode="after_pairs",
        max_length=20,
        num_episodes=5,
        batch_size=2,
        deterministic_vocab=True,
        full_universe_vocab=True,
        randomize_pairs=True,
        include_pass_token=True
    )
    
    # Test getting a batch
    batch = next(iter(dataloader))
    s, a, rtg, d, timesteps, mask = batch
    
    print(f"Batch shapes:")
    print(f"  Observations: {s.shape}")
    print(f"  Targets: {a.shape}")
    print(f"  RTG: {rtg.shape}")
    print(f"  Done: {d.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    print(f"  Mask: {mask.shape}")
    
    print("ARShot dataset restructuring complete!")
