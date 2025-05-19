import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class ViZDoomIterDataset(Dataset):
    """A PyTorch Dataset for loading and processing ViZDoom environment trajectories.
    
    This dataset handles loading and preprocessing of ViZDoom environment trajectories
    stored as numpy files. It processes observations, actions, and rewards from
    the ViZDoom environment, with support for trajectory filtering and normalization.

    Attributes:
        directory (str): Path to the directory containing trajectory data files.
        file_list (list): List of all data files in the directory.
        gamma (float): Discount factor for computing return-to-go (RTG).
        max_length (int): Maximum sequence length for trajectory segments.
            Note: Maximum possible value in the dataset is 1001.
        normalize (int): Whether to normalize observations (1 for normalization).
        filtered_list (list): List of filtered file names after processing.

    Note:
        The dataset expects each trajectory file to contain numpy arrays with keys:
        - 'obs': Observations (images) of shape (T, H, W, C)
        - 'action': Discrete actions
        - 'reward': Reward signals
        - 'done': Episode termination flags

        Special handling:
        - Observations are optionally normalized to [0, 1] range by dividing by 255
        - Only trajectories of exact length max_length are included
        - All tensors are trimmed to max_length from the beginning
    """

    def __init__(self, directory: str, gamma: float, max_length: int, normalize: int):
        """Initialize the ViZDoom dataset.

        Args:
            directory: Path to the directory containing trajectory data files.
            gamma: Discount factor for computing return-to-go values.
            max_length: Maximum number of timesteps to use in each trajectory segment.
                Note: Maximum possible value in the dataset is 1001.
            normalize: If 1, normalizes observations to [0, 1] range by dividing by 255.
        """
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.gamma = gamma
        self.max_length = max_length
        self.normalize = normalize
        self.filtered_list = []
        print('Filtering data...')
        self.filter_trajectories()

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
    
    def filter_trajectories(self) -> None:
        """Filter trajectory files based on length criteria.

        This method processes all trajectory files in the directory and stores
        only those that have an exact length of max_length in filtered_list.
        This ensures that all trajectories have consistent length during loading.

        Note:
            The filtering is done by checking the observation sequence length
            in each trajectory file.
        """
        for idx in tqdm(range(len(self.file_list))):
            file_path = os.path.join(self.directory, self.file_list[idx])
            data = np.load(file_path)
            if data['obs'].shape[0] == self.max_length:
                self.filtered_list.append(self.file_list[idx])

    def __len__(self) -> int:
        """Get the total number of available trajectory files.

        Returns:
            The number of trajectory files available in the dataset after filtering.
        """
        return len(self.filtered_list)

    def __getitem__(self, idx: int) -> tuple:
        """Get a trajectory segment by index.

        Args:
            idx: Index of the trajectory to retrieve.

        Returns:
            A tuple containing:
                - s (torch.Tensor): Observations of shape (max_length, C, H, W)
                - a (torch.Tensor): Actions of shape (max_length,)
                - rtg (torch.Tensor): Return-to-go values of shape (max_length,)
                - d (torch.Tensor): Done flags of shape (max_length,)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length,)
                - mask (torch.Tensor): Mask tensor of shape (max_length,)

        Note:
            - Observations are optionally normalized to [0, 1] range
            - All tensors are trimmed to max_length from the beginning
            - Observations are converted from (H, W, C) to (C, H, W) format
            - Mask is a tensor of ones since all trajectories have exact length
        """
        file_path = os.path.join(self.directory, self.filtered_list[idx])
        data = np.load(file_path)

        s = data['obs']
        a = data['action']
        r = data['reward']
        d = data['done']
        
        s = torch.from_numpy(s).float()

        if self.normalize == 1:
            s = s / 255.0
        
        s = s.unsqueeze(0)

        a = torch.from_numpy(a).unsqueeze(0).unsqueeze(-1)
        rtg = torch.from_numpy(self.discount_cumsum(r)).unsqueeze(0).unsqueeze(-1)
        d = torch.from_numpy(d).unsqueeze(0).unsqueeze(-1).to(dtype=torch.long)
       
        timesteps = torch.from_numpy(np.arange(0, self.max_length).reshape(1, -1, 1))
        mask = torch.ones_like(a)
        
        # * from beginning of trajectory
        s = s[:, :self.max_length, :, :, :]
        a = a[:, :self.max_length, :]
        rtg = rtg[:, :self.max_length, :]
        d = d[:, :self.max_length, :]
        mask = mask[:, :self.max_length, :]

        return s.squeeze(0), a.squeeze(0), rtg.squeeze(0), d.squeeze(), timesteps.squeeze(), mask.squeeze()