"""
ARShot Dataset Class

A PyTorch Dataset for loading and processing ARShot environment trajectories.
Compatible with the existing RATE/MATL training pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import pickle
from typing import Tuple, Optional, Dict, Any, List
import torch.nn.functional as F
from tqdm import tqdm
from src.envs.associative_retrieval.arshotenv import ARShotEnv


class ARShotDataset(Dataset):
    """A PyTorch Dataset for generating and processing ARShot environment trajectories.
    
    This dataset handles generating ARShot environment trajectories in-memory.
    It processes observations (tokens), actions (tokens), and rewards from the
    associative retrieval environment with 'shot' queries.

    Attributes:
        n_pairs (int): Number of key-value pairs in the environment.
        shot_mode (str): Mode for shot placement ("after_pairs" or "after_any_colon").
        max_length (int): Maximum sequence length for trajectory segments.
        gamma (float): Discount factor for computing return-to-go (RTG).
        num_episodes (int): Number of episodes to generate.
        env_kwargs (dict): Additional keyword arguments for environment creation.
        episodes (list): Generated episodes stored in memory.

    Note:
        The dataset generates episodes with the following structure:
        - 'obs': Token strings (not indices)
        - 'action': Action token strings  
        - 'reward': Scalar reward values
        - 'done': Episode termination flags
        
        Episodes are padded to max_length if shorter than max_length.
        Padding uses special values:
        - Actions are padded with -10 (special padding index)
        - Rewards are padded with 0.0
        - Done flags are padded with False
    """

    def __init__(self, n_pairs: int = 6, shot_mode: str = "after_pairs", 
                 max_length: int = 100, gamma: float = 1.0, num_episodes: int = 1000,
                 data_dir: str = "data/ARShot", **env_kwargs):
        """Initialize the ARShot dataset.

        Args:
            n_pairs: Number of key-value pairs in the environment.
            shot_mode: Mode for shot placement ("after_pairs" or "after_any_colon").
            max_length: Maximum number of timesteps to use in each trajectory segment.
            gamma: Discount factor for computing return-to-go values.
            num_episodes: Number of episodes to generate and store in memory.
            data_dir: Directory to store datasets.
            **env_kwargs: Additional keyword arguments passed to ARShotEnv.
        """
        self.n_pairs = n_pairs
        self.shot_mode = shot_mode
        self.max_length = max_length
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.env_kwargs = env_kwargs
        self.data_dir = data_dir
        
        # Generate dataset filename based on configuration
        self.dataset_filename = self._generate_filename()
        self.dataset_path = os.path.join(data_dir, self.dataset_filename)
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Try to load existing dataset, otherwise generate new one
        if self._load_dataset():
            print(f"Loaded existing ARShot dataset: {self.dataset_path}")
        else:
            print(f"Generating new ARShot dataset...")
            self.episodes = []
            self._generate_episodes()
            self._save_dataset()
            print(f"Saved ARShot dataset: {self.dataset_path}")
        
        # Create a reference environment to get vocab mappings
        self.env = ARShotEnv(n_pairs=n_pairs, shot_mode=shot_mode, **env_kwargs)
        self.vocab_size = len(self.env.vocab)
        
        print(f"ARShot dataset ready:")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  n_pairs: {n_pairs}")
        print(f"  shot_mode: {shot_mode}")
        print(f"  max_length: {max_length}")
        print(f"  vocab_size: {self.vocab_size}")
        print(f"  dataset_file: {self.dataset_filename}")

    def _generate_episodes(self):
        """Generate episodes and store them in memory."""
        for episode_idx in tqdm(range(self.num_episodes)):
            # Create a fresh environment for each episode with a different seed
            env = ARShotEnv(
                n_pairs=self.n_pairs, 
                shot_mode=self.shot_mode,
                rng_seed=episode_idx + 1000,  # Use episode index as seed for reproducibility
                **self.env_kwargs
            )
            
            obs, info = env.reset()
            done = False
            t = 0
            
            episode_data = {
                'obs': [],
                'action': [],
                'reward': [],
                'done': [],
                'info': info
            }
            
            # Add initial observation
            obs_token = env.id_to_token[obs]
            episode_data['obs'].append(obs_token)
            
            while not done and t < self.max_length:
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
                
                episode_data['action'].append(act_token)
                episode_data['reward'].append(reward)
                episode_data['done'].append(done or truncated)
                
                # Add next observation if not done
                if not (done or truncated) and t + 1 < self.max_length:
                    next_obs_token = env.id_to_token[next_obs]
                    episode_data['obs'].append(next_obs_token)
                
                obs = next_obs
                t += 1
            
            # Pad episode to max_length if needed
            current_length = len(episode_data['obs'])
            if current_length < self.max_length:
                pad_length = self.max_length - current_length
                
                # Pad observations with 'pass'
                episode_data['obs'].extend(['pass'] * pad_length)
                
                # Pad actions with -10 (but we need one less action than observations)
                if len(episode_data['action']) < self.max_length:
                    action_pad_length = self.max_length - len(episode_data['action'])
                    # episode_data['action'].extend(['pass'] * action_pad_length)
                    episode_data['action'].extend([-10] * action_pad_length)
                
                # Pad rewards and dones
                episode_data['reward'].extend([0.0] * pad_length)
                episode_data['done'].extend([False] * pad_length)
            
            # Ensure we have the right lengths
            if len(episode_data['obs']) > self.max_length:
                episode_data['obs'] = episode_data['obs'][:self.max_length]
            if len(episode_data['action']) > self.max_length:
                episode_data['action'] = episode_data['action'][:self.max_length]
            if len(episode_data['reward']) > self.max_length:
                episode_data['reward'] = episode_data['reward'][:self.max_length]
            if len(episode_data['done']) > self.max_length:
                episode_data['done'] = episode_data['done'][:self.max_length]
            
            self.episodes.append(episode_data)

    def _generate_filename(self) -> str:
        """Generate a filename based on dataset configuration.
        
        Returns:
            A descriptive filename for the dataset.
        """
        # Include key parameters that affect the dataset
        max_vocab = self.env_kwargs.get('max_vocab_size', 'unlimited')
        randomize = self.env_kwargs.get('randomize_pairs', True)
        
        filename = f"arshot_n{self.n_pairs}_{self.shot_mode}_ep{self.num_episodes}_len{self.max_length}"
        filename += f"_vocab{max_vocab}_rand{randomize}.pkl"
        
        return filename

    def _save_dataset(self) -> None:
        """Save the generated episodes to file."""
        try:
            dataset_data = {
                'episodes': self.episodes,
                'config': {
                    'n_pairs': self.n_pairs,
                    'shot_mode': self.shot_mode,
                    'max_length': self.max_length,
                    'num_episodes': self.num_episodes,
                    'gamma': self.gamma,
                    'env_kwargs': self.env_kwargs
                }
            }
            
            with open(self.dataset_path, 'wb') as f:
                pickle.dump(dataset_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            print(f"  Saved {len(self.episodes)} episodes")
            
        except Exception as e:
            print(f"  Warning: Failed to save dataset to {self.dataset_path}: {e}")

    def _load_dataset(self) -> bool:
        """Load episodes from dataset file if it exists and matches configuration.
        
        Returns:
            True if successfully loaded, False otherwise.
        """
        if not os.path.exists(self.dataset_path):
            return False
            
        try:
            with open(self.dataset_path, 'rb') as f:
                dataset_data = pickle.load(f)
            
            # Verify that dataset matches current configuration
            config = dataset_data.get('config', {})
            if (config.get('n_pairs') != self.n_pairs or
                config.get('shot_mode') != self.shot_mode or
                config.get('max_length') != self.max_length or
                config.get('num_episodes') != self.num_episodes):
                print(f"  Dataset configuration mismatch, regenerating...")
                return False
            
            # Load episodes
            self.episodes = dataset_data['episodes']
            print(f"  Loaded {len(self.episodes)} episodes")
            return True
            
        except Exception as e:
            print(f"  Warning: Failed to load dataset from {self.dataset_path}: {e}")
            return False

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

    def __len__(self) -> int:
        """Get the total number of available episodes.

        Returns:
            The number of episodes available in the dataset.
        """
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                           torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an episode by index.

        Args:
            idx: Index of the episode to retrieve.

        Returns:
            A tuple containing:
                - s (torch.Tensor): Observations as token indices of shape (max_length,)
                - a (torch.Tensor): Actions as token indices of shape (max_length,)
                - rtg (torch.Tensor): Return-to-go values of shape (max_length, 1)
                - d (torch.Tensor): Done flags of shape (max_length,)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length,)
                - mask (torch.Tensor): Mask tensor of shape (max_length,)
        """
        episode = self.episodes[idx]
        
        # Convert tokens to indices using the reference environment
        obs_indices = [self.env.token_to_id.get(token, self.env.token_to_id['pass']) 
                      for token in episode['obs']]
        # action_indices = [self.env.token_to_id.get(token, self.env.token_to_id['pass']) 
        #                  for token in episode['action']]
        # Handle action indices, with -10 as padding
        action_indices = []
        for token in episode['action']:
            if token == -10:
                action_indices.append(-10)  # Keep padding value as -10
            else:
                action_indices.append(self.env.token_to_id.get(token, self.env.token_to_id['pass']))
        
        # Convert to tensors
        s = torch.tensor(obs_indices, dtype=torch.long)
        a = torch.tensor(action_indices, dtype=torch.long)
        r = torch.tensor(episode['reward'], dtype=torch.float)
        d = torch.tensor(episode['done'], dtype=torch.long)
        
        # Compute return-to-go
        rtg = torch.from_numpy(self.discount_cumsum(r.numpy())).float()
        
        # Create timesteps
        timesteps = torch.arange(len(s), dtype=torch.long)
        
        # Create mask (1 for valid timesteps, 0 for padding)
        mask = torch.ones(len(s), dtype=torch.float)
        
        # Find where padding starts (where we have 'pass' tokens that are padding)
        # This is a bit tricky since 'pass' can be a legitimate action too
        # We'll use the done flags to determine valid timesteps
        for i, done_flag in enumerate(episode['done']):
            if done_flag and i < len(s) - 1:
                # Mark subsequent timesteps as padding
                mask[i+1:] = 0
                break
        
        return s.unsqueeze(-1), a.unsqueeze(-1), rtg.unsqueeze(-1), d.unsqueeze(-1), timesteps.unsqueeze(-1), mask.unsqueeze(-1)


def create_arshot_dataloader(n_pairs: int = 6, shot_mode: str = "after_pairs",
                            max_length: int = 100, gamma: float = 1.0, 
                            num_episodes: int = 1000, batch_size: int = 32,
                            shuffle: bool = True, num_workers: int = 0,
                            data_dir: str = "data/ARShot",
                            **env_kwargs) -> DataLoader:
    """Create a DataLoader for ARShot datasets.
    
    Args:
        n_pairs: Number of key-value pairs in the environment.
        shot_mode: Mode for shot placement ("after_pairs" or "after_any_colon").
        max_length: Maximum sequence length.
        gamma: Discount factor for return-to-go computation.
        num_episodes: Number of episodes to generate.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        data_dir: Directory to store datasets.
        **env_kwargs: Additional environment kwargs.
        
    Returns:
        A PyTorch DataLoader for the ARShot dataset.
    """
    dataset = ARShotDataset(
        n_pairs=n_pairs,
        shot_mode=shot_mode,
        max_length=max_length,
        gamma=gamma,
        num_episodes=num_episodes,
        data_dir=data_dir,
        **env_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def list_datasets(data_dir: str = "data/ARShot") -> None:
    """List all available ARShot datasets.
    
    Args:
        data_dir: Directory containing datasets.
    """
    if not os.path.exists(data_dir):
        print(f"No datasets found. Directory {data_dir} does not exist.")
        return
    
    datasets = [f for f in os.listdir(data_dir) if f.startswith("arshot_") and f.endswith(".pkl")]
    
    if not datasets:
        print(f"No ARShot datasets found in {data_dir}")
        return
    
    print(f"Found {len(datasets)} ARShot datasets in {data_dir}:")
    for dataset in sorted(datasets):
        print(f"  {dataset}")

def clear_datasets(data_dir: str = "data/ARShot") -> None:
    """Clear all ARShot datasets.
    
    Args:
        data_dir: Directory containing datasets.
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return
        
    datasets = [f for f in os.listdir(data_dir) if f.startswith("arshot_") and f.endswith(".pkl")]
    
    if not datasets:
        print(f"No ARShot datasets found in {data_dir}")
        return
    
    for dataset in datasets:
        os.remove(os.path.join(data_dir, dataset))
    
    print(f"Removed {len(datasets)} datasets from {data_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARShot Dataset utilities")
    parser.add_argument("--test", action="store_true", help="Run dataset test")
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--clear", action="store_true", help="Clear all datasets")
    parser.add_argument("--data-dir", type=str, default="data/ARShot", help="Data directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets(args.data_dir)
    elif args.clear:
        clear_datasets(args.data_dir)
    elif args.test:
        # Test the dataset
        print("Testing ARShot dataset...")
        dataset = ARShotDataset(
            n_pairs=3,
            shot_mode="after_pairs",
            max_length=20,
            num_episodes=10,
            deterministic_vocab=True,
            full_universe_vocab=True,
            randomize_pairs=True,
            include_pass_token=True
        )
        
        # Test getting an episode
        s, a, rtg, d, timesteps, mask = dataset[0]
        print(f"Episode 0:")
        print(f"  Observations shape: {s.shape}")
        print(f"  Actions shape: {a.shape}")
        print(f"  RTG shape: {rtg.shape}")
        print(f"  Done shape: {d.shape}")
        print(f"  Timesteps shape: {timesteps.shape}")
        print(f"  Mask shape: {mask.shape}")
        
        # Show first few tokens
        cutoff = 20
        print(f"  First 10 obs tokens: {[dataset.env.id_to_token[idx.item()] for idx in s[:cutoff]]}")
        print(f"  First 10 action tokens: {[dataset.env.id_to_token[idx.item()] for idx in a[:cutoff]]}")
        print(f"  First 10 rewards: {rtg[:cutoff, 0].tolist()}")
        print(f"  First 10 done: {d[:cutoff].tolist()}")
        print(f"  First 10 timesteps: {timesteps[:cutoff].tolist()}")
        print(f"  First 10 mask: {mask[:cutoff].tolist()}")
    else:
        print("Usage: python -m src.envs_datasets.arshot_dataset [--test] [--list] [--clear]")
