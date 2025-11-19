import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm


def discount_cumsum(x, gamma):
    """Compute discounted cumulative sum of rewards."""
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


class ViZDoomTrajectoryDataset(Dataset):
    """Dataset for loading ViZDoom trajectories for Elastic-DT training.
    
    This dataset loads ViZDoom trajectories from .npz files and formats them
    for training the Elastic Decision Transformer model.
    """

    def __init__(self, dataset_dir, context_len, rtg_scale, normalize=True):
        """
        Args:
            dataset_dir: Directory containing .npz trajectory files
            context_len: Length of context window (K in Decision Transformer)
            rtg_scale: Scale factor for returns-to-go
            normalize: Whether to normalize images to [0, 1]
        """
        self.context_len = context_len
        self.rtg_scale = rtg_scale
        self.normalize = normalize
        self.dataset_dir = dataset_dir
        
        # Load all trajectory files
        print(f"Loading trajectories from {dataset_dir}")
        self.file_list = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
        self.trajectories = []
        
        print("Loading and preprocessing trajectories...")
        for filename in tqdm(self.file_list):
            file_path = os.path.join(dataset_dir, filename)
            data = np.load(file_path)
            
            # Extract data
            obs = data['obs']  # Shape: (T, C, W, H) from ViZDoom env
            actions = data['action']  # Shape: (T,)
            rewards = data['reward']  # Shape: (T,)
            dones = data['done']  # Shape: (T,)
            
            # Convert observations from (T, C, W, H) to (T, C, H, W) for PyTorch
            # ViZDoom returns (C, W, H) = (3, 112, 64), need (C, H, W) = (3, 64, 112)
            obs = obs.transpose(0, 1, 3, 2)  # Now: (T, C, H, W)
            
            # Compute returns-to-go
            rtg = discount_cumsum(rewards, gamma=1.0)
            
            traj = {
                'observations': obs.astype(np.float32),
                'actions': actions.astype(np.int64),
                'rewards': rewards.astype(np.float32),
                'returns_to_go': rtg.astype(np.float32),
                'dones': dones.astype(np.bool_)
            }
            
            self.trajectories.append(traj)
        
        print(f"Loaded {len(self.trajectories)} trajectories")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # Sample random starting index
            si = random.randint(0, traj_len - self.context_len)
            
            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            rewards = torch.from_numpy(traj['rewards'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(
                traj['returns_to_go'][si : si + self.context_len]
            ) / self.rtg_scale
            
            # Normalize images to [0, 1] if needed
            if self.normalize:
                states = states / 255.0
            
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            # Pad trajectory if it's shorter than context_len
            padding_len = self.context_len - traj_len
            
            states = torch.from_numpy(traj['observations'])
            if self.normalize:
                states = states / 255.0
            
            # Pad states (C, H, W format)
            states = torch.cat([
                states,
                torch.zeros((padding_len, *states.shape[1:]), dtype=states.dtype)
            ], dim=0)
            
            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([
                actions,
                torch.zeros(padding_len, dtype=actions.dtype)
            ], dim=0)
            
            rewards = torch.from_numpy(traj['rewards'])
            rewards = torch.cat([
                rewards,
                torch.zeros(padding_len, dtype=rewards.dtype)
            ], dim=0)
            
            returns_to_go = torch.from_numpy(traj['returns_to_go']) / self.rtg_scale
            returns_to_go = torch.cat([
                returns_to_go,
                torch.zeros(padding_len, dtype=returns_to_go.dtype)
            ], dim=0)
            
            timesteps = torch.arange(start=0, end=self.context_len, step=1)
            
            traj_mask = torch.cat([
                torch.ones(traj_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ], dim=0)

        return (
            timesteps,
            states,
            actions,
            returns_to_go.unsqueeze(-1),
            rewards.unsqueeze(-1),
            traj_mask
        )

