import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
from src.utils.additional_data_processors import coords_to_idx, idx_to_coords


class POPGymDataset(Dataset):
    """A PyTorch Dataset for loading and processing POPGym environment trajectories.
    
    This dataset handles loading and preprocessing of POPGym environment trajectories
    stored as numpy files. It processes observations, actions, and rewards from various
    POPGym environments including Battleship and Minesweeper variants.

    Attributes:
        directory (str): Path to the directory containing trajectory data files.
        file_list (list): List of all data files in the directory.
        gamma (float): Discount factor for computing return-to-go (RTG).
        max_length (int): Maximum sequence length for trajectory segments.
        env_name (str): Name of the POPGym environment (e.g., 'BattleshipEasy', 'MineSweeperMedium').

    Note:
        The dataset expects each trajectory file to contain numpy arrays with keys:
        - 'obs': Observations
        - 'action': Actions (coordinates for Battleship/Minesweeper, continuous for Pendulum)
        - 'reward': Reward signals
        - 'done': Episode termination flags

        Special handling:
        - Actions are converted from coordinates to indices for Battleship/Minesweeper
        - Board sizes vary by environment:
            - Battleship: Easy (8x8), Medium (10x10), Hard (12x12)
            - Minesweeper: Easy (4x4), Medium (6x6), Hard (8x8)
        - Shorter trajectories are padded to max_length
        - Padding uses special values for different tensors
    """

    def __init__(self, directory: str, gamma: float, max_length: int, env_name: str):
        """Initialize the POPGym dataset.

        Args:
            directory: Path to the directory containing trajectory data files.
            gamma: Discount factor for computing return-to-go values.
            max_length: Maximum number of timesteps to use in each trajectory segment.
            env_name: Name of the POPGym environment. Must be one of:
                - 'BattleshipEasy', 'BattleshipMedium', 'BattleshipHard'
                - 'MineSweeperEasy', 'MineSweeperMedium', 'MineSweeperHard'
                - 'Pendulum'
        """
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.gamma = gamma
        self.max_length = max_length
        self.env_name = env_name

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
        """Get the total number of available trajectory files.

        Returns:
            The number of trajectory files available in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
        """Get a trajectory segment by index.

        Args:
            idx: Index of the trajectory to retrieve.

        Returns:
            A tuple containing:
                - s (torch.Tensor): Observations
                - a (torch.Tensor): Actions (indices for Battleship/Minesweeper, continuous for Pendulum)
                - rtg (torch.Tensor): Return-to-go values of shape (max_length, 1)
                - d (torch.Tensor): Done flags of shape (max_length,)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length,)
                - mask (torch.Tensor): Mask tensor of shape (max_length,)

        Note:
            - Actions are converted based on environment:
                - Battleship/Minesweeper: 2D coordinates converted to 1D indices
                - Pendulum: Continuous actions preserved as is
            - Shorter trajectories are padded to max_length:
                - Observations are padded with the last state
                - Actions are padded with -10
                - Done flags are padded with 2
                - Masks are padded with 0
                - Return-to-go values are padded with 0
            - Tensor shapes are adjusted based on environment type
        """
        file_path = os.path.join(self.directory, self.file_list[idx])
        data = np.load(file_path)
        s = data['obs']
        a = data['action']
        r = data['reward']
        d = data['done']

        length = s.shape[0]
        
        s = torch.from_numpy(s).float().unsqueeze(0)
        a = torch.from_numpy(a).unsqueeze(0).unsqueeze(-1)
        rtg = torch.from_numpy(self.discount_cumsum(r)).unsqueeze(0).unsqueeze(-1)
        d = torch.from_numpy(d).unsqueeze(0).unsqueeze(-1).to(dtype=torch.long)
        timesteps = torch.from_numpy(np.arange(0, length).reshape(1, -1, 1))
        mask = torch.ones_like(a)

        if 'BattleshipEasy' in self.env_name:
            a = coords_to_idx(a.squeeze(-1), board_size=8)
        elif 'BattleshipMedium' in self.env_name:
            a = coords_to_idx(a.squeeze(-1), board_size=10)
        elif 'BattleshipHard' in self.env_name:
            a = coords_to_idx(a.squeeze(-1), board_size=12)
        elif 'MineSweeperEasy' in self.env_name:
            a = coords_to_idx(a.squeeze(-1), board_size=4)
        elif 'MineSweeperMedium' in self.env_name:
            a = coords_to_idx(a.squeeze(-1), board_size=6)
        elif 'MineSweeperHard' in self.env_name:
            a = coords_to_idx(a.squeeze(-1), board_size=8)
        elif 'Pendulum' in self.env_name:
            a = a.squeeze(-1)

        #Pad with zeros if length is less than max_length
        if length <= self.max_length:
            pad_length = self.max_length - length
            state_to_pad = s[:, -1, :].unsqueeze(1)
            while s.shape[1] < self.max_length:
                s = torch.cat((s, state_to_pad), dim=1)

            mask_to_pad = mask[:, -1, :].unsqueeze(1)
            while mask.shape[1] < self.max_length:
                mask = torch.cat((mask, mask_to_pad), dim=1)
            a = F.pad(a, (0, 0, 0, pad_length+1, 0, 0), value=-10)
            d = F.pad(d, (0, 0, 0, pad_length, 0, 0), value=2)
            timesteps = F.pad(timesteps, (0, 0, 0, pad_length, 0, 0), value=0)
            rtg = F.pad(rtg, (0, 0, 0, pad_length, 0, 0), value=0)

        # * from beginning of trajectory
        s = s[:, :self.max_length, :]
        a = a[:, :self.max_length, :]
        rtg = rtg[:, :self.max_length, :]
        d = d[:, :self.max_length, :]
        timesteps = timesteps[:, :self.max_length, :]
        mask = mask[:, :self.max_length, :]

        s = s.squeeze(0)
        a = a.squeeze(0)
        rtg = rtg.squeeze(0)
        d = d.squeeze()
        timesteps = timesteps.squeeze()
        mask = mask.squeeze()

        if len(s.shape) == 2:
            pass
        elif len(s.shape) == 3:
            s = s.squeeze(-2)
        else:
            raise ValueError('State length is not 2 or 3, this is unexpected for POPGym')
        
        if len(a.shape) == 2:
            pass
        elif len(a.shape) == 3:
            a = a.squeeze(-1)
        else:
            raise ValueError('Action length is not 2 or 3, this is unexpected for POPGym')
        
        
        return s, a, rtg, d, timesteps, mask