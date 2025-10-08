"""
MDP Dataset Class

A PyTorch Dataset for loading and processing MDP environment trajectories.
Compatible with the existing RATE/ELMUR training pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any
import gymnasium as gym


class MDPDataset(Dataset):
    """A PyTorch Dataset for loading and processing MDP environment trajectories.
    
    This dataset handles loading and preprocessing of MDP environment trajectories
    stored as numpy files. It processes observations, actions, and rewards from
    classic control environments like CartPole, MountainCar, Acrobot, and Pendulum.

    Attributes:
        directory (str): Path to the directory containing trajectory data files.
        file_list (list): List of all data files in the directory.
        gamma (float): Discount factor for computing return-to-go (RTG).
        max_length (int): Maximum sequence length for trajectory segments.
        env_name (str): Name of the MDP environment.
        action_space_type (str): Type of action space ('discrete' or 'continuous').
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.

    Note:
        The dataset expects each trajectory file to contain numpy arrays with keys:
        - 'obs': Observations (state vectors)
        - 'action': Actions (discrete indices or continuous values)
        - 'reward': Reward signals
        - 'done': Episode termination flags

        Special handling:
        - For long episodes (>= max_length), random cropping is used to extract segments
        - Shorter trajectories are padded to max_length
        - Padding uses special values for different tensors
        - Actions are handled differently for discrete vs continuous spaces
    """

    def __init__(self, directory: str, gamma: float, max_length: int, env_name: str):
        """Initialize the MDP dataset.

        Args:
            directory: Path to the directory containing trajectory data files.
            gamma: Discount factor for computing return-to-go values.
            max_length: Maximum number of timesteps to use in each trajectory segment.
            env_name: Name of the MDP environment. Must be one of:
                - 'CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0'
                - 'Acrobot-v1', 'Pendulum-v1'
        """
        self.directory = directory
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.npz')]
        self.gamma = gamma
        self.max_length = max_length
        self.env_name = env_name
        
        # Load metadata if available
        metadata_path = os.path.join(directory, "dataset_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Determine action space type and dimensions
        self._setup_space_info()
        
        # Pre-compute trajectory segments for efficient random access
        self._build_segment_index()
        
        print(f"Loaded MDP dataset for {env_name}")
        print(f"  Files: {len(self.file_list)}")
        print(f"  Total segments: {len(self.segment_index)}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Action space: {self.action_space_type}")

    def _setup_space_info(self):
        """Setup information about state and action spaces."""
        # Create environment to get space information
        env = gym.make(self.env_name)
        
        # State space info
        if hasattr(env.observation_space, 'shape'):
            self.state_dim = env.observation_space.shape[0] if env.observation_space.shape else 1
        else:
            self.state_dim = 1
            
        # Action space info
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space_type = 'discrete'
            self.action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_space_type = 'continuous'
            self.action_dim = env.action_space.shape[0] if env.action_space.shape else 1
        else:
            raise ValueError(f"Unsupported action space type: {type(env.action_space)}")
            
        env.close()

    def _build_segment_index(self):
        """Build an index of all possible trajectory segments."""
        self.segment_index = []
        
        for file_idx, filename in enumerate(self.file_list):
            file_path = os.path.join(self.directory, filename)
            data = np.load(file_path)
            episode_length = data['obs'].shape[0]
            
            if episode_length >= self.max_length:
                # For long episodes, create multiple segments
                num_segments = episode_length - self.max_length + 1
                for start_idx in range(num_segments):
                    self.segment_index.append((file_idx, start_idx, self.max_length))
            else:
                # For short episodes, use the whole episode (will be padded)
                self.segment_index.append((file_idx, 0, episode_length))

            # if file_idx == 50: # TODO: remove after debugging
            #     break

    def discount_cumsum(self, x: np.ndarray) -> np.ndarray:
        """Compute the discounted cumulative sum of rewards.

        This method implements the standard discounted sum calculation:
        G_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ... + γ^{T-t} * r_T

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
        """Get the total number of available trajectory segments.

        Returns:
            The number of trajectory segments available in the dataset.
        """
        return len(self.segment_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                           torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a trajectory segment by index.

        Args:
            idx: Index of the trajectory segment to retrieve.

        Returns:
            A tuple containing:
                - s (torch.Tensor): Observations/states of shape (max_length, state_dim)
                - a (torch.Tensor): Actions of shape (max_length, 1) for discrete or (max_length, action_dim) for continuous
                - rtg (torch.Tensor): Return-to-go values of shape (max_length, 1)
                - d (torch.Tensor): Done flags of shape (max_length,)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length,)
                - mask (torch.Tensor): Mask tensor of shape (max_length,)

        Note:
            - For episodes >= max_length: Random cropping is used to extract segments
            - For episodes < max_length: Padding is applied
            - Padding strategy:
                - Observations are padded with the last state
                - Actions are padded with -10 (discrete) or zeros (continuous)
                - Done flags are padded with 2
                - Masks are padded with 0
                - Return-to-go values are padded with 0
            - All tensors are converted to appropriate torch dtypes
        """
        file_idx, start_idx, segment_length = self.segment_index[idx]
        file_path = os.path.join(self.directory, self.file_list[file_idx])
        data = np.load(file_path)
        
        # Extract the segment
        end_idx = start_idx + segment_length
        s = data['obs'][start_idx:end_idx]      # States/observations
        a = data['action'][start_idx:end_idx]   # Actions
        r = data['reward'][start_idx:end_idx]   # Rewards
        d = data['done'][start_idx:end_idx]     # Done flags

        length = s.shape[0]
        
        # Convert to torch tensors
        s = torch.from_numpy(s).float()
        r = torch.from_numpy(r).float()
        d = torch.from_numpy(d).long()
        
        # Handle actions based on action space type
        if self.action_space_type == 'discrete':
            # Discrete actions - convert to long tensor
            a = torch.from_numpy(a).long()
            if len(a.shape) == 1:
                a = a.unsqueeze(-1)  # Add action dimension
        else:
            # Continuous actions - keep as float
            a = torch.from_numpy(a).float()
            if len(a.shape) == 1:
                a = a.unsqueeze(-1)  # Add action dimension

        # Compute return-to-go
        rtg = torch.from_numpy(self.discount_cumsum(r.numpy())).float().unsqueeze(-1)
        
        # Create timesteps (relative to the segment start)
        timesteps = torch.arange(start_idx, start_idx + length, dtype=torch.long)
        
        # Create initial mask (all ones for valid timesteps)
        mask = torch.ones(length, dtype=torch.float)
        
        # Pad sequences to max_length if needed (only for short episodes)
        if length < self.max_length:
            pad_length = self.max_length - length
            
            # Pad states with last state
            last_state = s[-1:].expand(pad_length, -1)
            s = torch.cat([s, last_state], dim=0)
            
            # Pad actions
            if self.action_space_type == 'discrete':
                # Pad discrete actions with -10
                pad_actions = torch.full((pad_length, a.shape[-1]), -10, dtype=torch.long)
            else:
                # Pad continuous actions with zeros
                pad_actions = torch.zeros((pad_length, a.shape[-1]), dtype=torch.float)
            a = torch.cat([a, pad_actions], dim=0)
            
            # Pad return-to-go with zeros
            pad_rtg = torch.zeros((pad_length, 1), dtype=torch.float)
            rtg = torch.cat([rtg, pad_rtg], dim=0)
            
            # Pad done flags with 2 (special padding value)
            pad_done = torch.full((pad_length,), 2, dtype=torch.long)
            d = torch.cat([d, pad_done], dim=0)
            
            # Pad timesteps with the last timestep value
            pad_timesteps = torch.full((pad_length,), timesteps[-1], dtype=torch.long)
            timesteps = torch.cat([timesteps, pad_timesteps], dim=0)
            
            # Pad mask with zeros
            pad_mask = torch.zeros(pad_length, dtype=torch.float)
            mask = torch.cat([mask, pad_mask], dim=0)
        
        return s, a, rtg, d, timesteps, mask


def create_mdp_dataloader(data_dir: str, env_name: str, batch_size: int = 32, 
                         max_length: int = 1000, gamma: float = 0.99,
                         shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader for MDP datasets.
    
    Args:
        data_dir: Directory containing the dataset files.
        env_name: Name of the MDP environment.
        batch_size: Batch size for the DataLoader.
        max_length: Maximum sequence length.
        gamma: Discount factor for return-to-go computation.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        
    Returns:
        A PyTorch DataLoader for the MDP dataset.
    """
    dataset = MDPDataset(
        directory=data_dir,
        gamma=gamma,
        max_length=max_length,
        env_name=env_name
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def analyze_mdp_dataset(data_dir: str, env_name: str) -> Dict[str, Any]:
    """Analyze an MDP dataset and return statistics.
    
    Args:
        data_dir: Directory containing the dataset files.
        env_name: Name of the MDP environment.
        
    Returns:
        Dictionary containing dataset analysis results.
    """
    dataset = MDPDataset(
        directory=data_dir,
        gamma=0.99,
        max_length=1000,
        env_name=env_name
    )
    
    print(f"\nAnalyzing MDP dataset: {env_name}")
    print(f"Dataset directory: {data_dir}")
    print("=" * 60)
    
    # Basic info
    print(f"Number of episodes: {len(dataset.file_list)}")
    print(f"Number of segments: {len(dataset)}")
    print(f"State dimension: {dataset.state_dim}")
    print(f"Action dimension: {dataset.action_dim}")
    print(f"Action space type: {dataset.action_space_type}")
    
    # Load metadata if available
    if dataset.metadata:
        stats = dataset.metadata.get('statistics', {})
        print(f"\nDataset Statistics:")
        print(f"  Total steps: {stats.get('total_steps', 'N/A')}")
        print(f"  Mean episode return: {stats.get('mean_return', 'N/A'):.3f} ± {stats.get('std_return', 'N/A'):.3f}")
        print(f"  Return range: [{stats.get('min_return', 'N/A'):.3f}, {stats.get('max_return', 'N/A'):.3f}]")
        print(f"  Mean episode length: {stats.get('mean_length', 'N/A'):.1f} ± {stats.get('std_length', 'N/A'):.1f}")
        print(f"  Length range: [{stats.get('min_length', 'N/A')}, {stats.get('max_length', 'N/A')}]")
        
        print(f"\nObservation Statistics:")
        obs_mean = stats.get('obs_mean', [])
        obs_std = stats.get('obs_std', [])
        obs_min = stats.get('obs_min', [])
        obs_max = stats.get('obs_max', [])
        
        for i in range(len(obs_mean)):
            print(f"  Dim {i}: mean={obs_mean[i]:.3f}, std={obs_std[i]:.3f}, "
                  f"range=[{obs_min[i]:.3f}, {obs_max[i]:.3f}]")
        
        print(f"\nAction Statistics:")
        if dataset.action_space_type == 'discrete':
            action_dist = stats.get('action_distribution', {})
            print(f"  Action distribution: {action_dist}")
            print(f"  Unique actions: {stats.get('n_unique_actions', 'N/A')}")
        else:
            action_mean = stats.get('action_mean', 'N/A')
            action_std = stats.get('action_std', 'N/A')
            action_min = stats.get('action_min', 'N/A')
            action_max = stats.get('action_max', 'N/A')
            print(f"  Mean: {action_mean}")
            print(f"  Std: {action_std}")
            print(f"  Range: [{action_min}, {action_max}]")
        
        print(f"\nReward Statistics:")
        print(f"  Mean: {stats.get('reward_mean', 'N/A'):.3f}")
        print(f"  Std: {stats.get('reward_std', 'N/A'):.3f}")
        print(f"  Range: [{stats.get('reward_min', 'N/A'):.3f}, {stats.get('reward_max', 'N/A'):.3f}]")
        print(f"  Median: {stats.get('reward_median', 'N/A'):.3f}")
    
    return dataset.metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze MDP datasets")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing the dataset")
    parser.add_argument("--env-name", type=str, required=True,
                       help="Name of the MDP environment")
    
    args = parser.parse_args()
    
    analyze_mdp_dataset(args.data_dir, args.env_name)
