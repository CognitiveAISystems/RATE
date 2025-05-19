import torch
import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class MemoryMazeDataset(Dataset):
    """A PyTorch Dataset for loading and processing Memory Maze environment trajectories.
    
    This dataset handles loading, filtering, and preprocessing of Memory Maze environment
    trajectories stored as numpy files. It supports both full trajectory loading and
    filtered loading based on reward conditions.

    Attributes:
        directory (str): Path to the directory containing trajectory data files.
        file_list (list): List of all data files in the directory.
        gamma (float): Discount factor for computing return-to-go (RTG).
        max_length (int): Maximum sequence length for trajectory segments.
        filtered_list (list): List of filtered file names when using reward filtering.
        list_of_start_indexes (list): List of starting indices for filtered trajectories.
        only_non_zero_rewards (bool): Whether to filter trajectories based on rewards.

    Note:
        The dataset expects each trajectory file to contain numpy arrays with keys:
        - 'obs': Observations (images)
        - 'action': Actions
        - 'reward': Rewards
    """

    def __init__(self, directory: str, gamma: float, max_length: int, only_non_zero_rewards: bool):
        """Initialize the Memory Maze dataset.

        Args:
            directory: Path to the directory containing trajectory data files.
            gamma: Discount factor for computing return-to-go values.
            max_length: Maximum number of timesteps to use in each trajectory segment.
                Note: Maximum possible value in the dataset is 1001.
            only_non_zero_rewards: If True, only includes trajectory segments that have
                a total reward of at least 2 within the first max_length timesteps.
        """
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.gamma = gamma
        self.max_length = max_length
        self.filtered_list = []
        self.list_of_start_indexes = []
        self.only_non_zero_rewards = only_non_zero_rewards
        if self.only_non_zero_rewards:
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
        """Filter trajectories based on reward conditions.

        This method processes all trajectory files and selects segments that have
        a total reward of at least 2 within max_length timesteps. The filtering
        process:
        1. Iterates through all trajectory files
        2. For each file, checks segments of length max_length starting every 90 timesteps
        3. Stores file names and starting indices for segments meeting the reward criterion

        Note:
            This method is only called if only_non_zero_rewards is True during initialization.
        """
        for idx in tqdm(range(len(self.file_list)), total=len(self.file_list)):
            file_path = os.path.join(self.directory, self.file_list[idx])
            data = np.load(file_path)
            
            for i in range(0, 500-self.max_length, 90):
                if data['reward'][i:i+self.max_length].sum() >= 2:
                    self.filtered_list.append(self.file_list[idx])
                    self.list_of_start_indexes.append(i)

    def __len__(self) -> int:
        """Get the total number of available trajectory segments.

        Returns:
            The number of trajectory segments available in the dataset.
            If only_non_zero_rewards is True, returns the number of filtered segments.
            Otherwise, returns the total number of trajectory files.
        """
        if self.only_non_zero_rewards:
            return len(self.filtered_list)
        else:
            return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
        """Get a trajectory segment by index.

        Args:
            idx: Index of the trajectory segment to retrieve.

        Returns:
            A tuple containing:
                - image (torch.Tensor): Observation images of shape (max_length, C, H, W)
                - action (torch.Tensor): Actions of shape (max_length, 1)
                - rtg (torch.Tensor): Return-to-go values of shape (max_length, 1)
                - done (torch.Tensor): Done flags of shape (max_length,)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length,)
                - mask (torch.Tensor): Mask tensor of shape (max_length,)

        Note:
            Images are normalized to [0, 1] range by dividing by 255.
            Actions are converted from one-hot to single integer values.
        """
        if self.only_non_zero_rewards:
            file_path = os.path.join(self.directory, self.filtered_list[idx])
            start_idx = self.list_of_start_indexes[idx]
        else:
            file_path = os.path.join(self.directory, self.file_list[idx])
        data = np.load(file_path)

        image = torch.from_numpy(data['obs']).permute(0, 3, 1, 2).float()
        action = torch.from_numpy(data['action'])
        rtg = torch.from_numpy(self.discount_cumsum(data['reward'])).unsqueeze(-1)
        timesteps = torch.from_numpy(np.arange(start_idx, start_idx+self.max_length).reshape(-1))
        done = torch.zeros_like(timesteps)
        done[-1] = 1
        mask = torch.ones_like(timesteps)
        
        image = image[start_idx:start_idx+self.max_length, :, :, :]
        image = image / 255.
        action = action[start_idx:start_idx+self.max_length, :]
        rtg = rtg[start_idx:start_idx+self.max_length, :]

        action = torch.argmax(action, dim=-1).unsqueeze(-1)

        return image, action, rtg, done, timesteps, mask