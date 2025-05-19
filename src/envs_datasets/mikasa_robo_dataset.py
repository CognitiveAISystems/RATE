import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class MIKASARoboIterDataset(Dataset):
    """A PyTorch Dataset for loading and processing MIKASA robot manipulation trajectories.
    
    This dataset handles loading and preprocessing of robot manipulation trajectories
    stored as numpy files. It processes RGB observations, actions, and success signals
    from robot manipulation tasks.

    Attributes:
        directory (str): Path to the directory containing trajectory data files.
        file_list (list): List of all data files in the directory.
        gamma (float): Discount factor for computing return-to-go (RTG).
        max_length (int): Maximum sequence length for trajectory segments.
        normalize (int): Whether to normalize RGB observations (1 for normalization).
        filtered_list (list): List of filtered file names after processing.

    Note:
        The dataset expects each trajectory file to contain numpy arrays with keys:
        - 'rgb': RGB observations of shape (T, H, W, C)
        - 'action': Robot actions
        - 'success': Success signals (used as rewards)
        - 'done': Episode termination flags
    """

    def __init__(self, directory: str, gamma: float, max_length: int, normalize: int):
        """Initialize the MIKASA robot dataset.

        Args:
            directory: Path to the directory containing trajectory data files.
            gamma: Discount factor for computing return-to-go values.
            max_length: Maximum number of timesteps to use in each trajectory segment.
            normalize: If 1, normalizes RGB observations to [0, 1] range by dividing by 255.
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
            x: 1D numpy array of reward values (success signals).

        Returns:
            A 1D numpy array containing the discounted cumulative sums for each timestep.
        """
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum
    
    def filter_trajectories(self) -> None:
        """Filter and process trajectory files.

        This method processes all trajectory files in the directory and stores
        their names in filtered_list. Currently, all files are included in the
        filtered list, but this method can be extended to implement additional
        filtering criteria.

        Note:
            The method uses memory-mapped loading (mmap_mode='r') for efficient
            file handling with large datasets.
        """
        # for idx in tqdm(range(100)):
        for idx in tqdm(range(len(self.file_list))):
            file_path = os.path.join(self.directory, self.file_list[idx])
            # data = np.load(file_path)
            # if data['rgb'].shape[0] <= self.max_length:
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
                - s (torch.Tensor): RGB observations of shape (max_length, C, H, W)
                - a (torch.Tensor): Actions of shape (max_length,)
                - rtg (torch.Tensor): Return-to-go values of shape (max_length, 1)
                - d (torch.Tensor): Done flags of shape (max_length,)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length,)
                - mask (torch.Tensor): Mask tensor of shape (max_length,)

        Note:
            - RGB observations are optionally normalized to [0, 1] range
            - All tensors are trimmed to max_length
            - Success signals are used as rewards for computing return-to-go
            - Data is loaded using memory mapping for efficiency
        """
        file_path = os.path.join(self.directory, self.filtered_list[idx])
        #print(file_path)
        # data = np.load(file_path)
        data = np.load(file_path, mmap_mode='r')

        s = data['rgb'][:self.max_length].transpose(0, 3, 1, 2)
        # j = data['joints']
        a = data['action']
        # r = data['reward']
        r = data['success']
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

        return s.squeeze(0), a.squeeze(0).squeeze(-1), rtg.squeeze(0), d.squeeze(), timesteps.squeeze(), mask.squeeze()