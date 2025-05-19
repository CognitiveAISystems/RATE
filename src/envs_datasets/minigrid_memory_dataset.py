import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F

class MinigridMemoryIterDataset(Dataset):
    """A PyTorch Dataset for loading and processing Minigrid Memory environment trajectories.
    
    This dataset handles loading and preprocessing of Minigrid Memory environment
    trajectories stored as numpy files. It processes observations, actions, and rewards
    from memory-based navigation tasks in the Minigrid environment.

    Attributes:
        directory (str): Path to the directory containing trajectory data files.
        file_list (list): List of all data files in the directory.
        gamma (float): Discount factor for computing return-to-go (RTG).
        max_length (int): Maximum sequence length for trajectory segments.
        normalize (int): Whether to normalize observations (1 for normalization).
        filtered_list (list): List of filtered file names after processing.

    Note:
        The dataset expects each trajectory file to contain numpy arrays with keys:
        - 'obs': Observations of shape (T, C, H, W) with values in [0, 1]
        - 'action': Discrete actions
        - 'reward': Reward signals
        - 'done': Episode termination flags

        Special handling:
        - Observations are scaled by 255 during loading (as they are stored normalized)
        - Shorter trajectories are padded to max_length
        - Padding uses the last state for observations and special values for other tensors
    """

    def __init__(self, directory: str, gamma: float, max_length: int, normalize: int):
        """Initialize the Minigrid Memory dataset.

        Args:
            directory: Path to the directory containing trajectory data files.
            gamma: Discount factor for computing return-to-go values.
            max_length: Maximum number of timesteps to use in each trajectory segment.
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
        only those that have a length less than or equal to max_length in
        filtered_list. This ensures that all trajectories can be properly
        padded to max_length during loading.

        Note:
            The filtering is done by checking the observation sequence length
            in each trajectory file.
        """
        for idx in tqdm(range(len(self.file_list))):
            #for idx in range(len(self.file_list)):
            file_path = os.path.join(self.directory, self.file_list[idx])
            data = np.load(file_path)
            # print(data['obs'].shape[0])
            if data['obs'].shape[0] <= self.max_length:
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
                - a (torch.Tensor): Actions of shape (max_length, 1)
                - rtg (torch.Tensor): Return-to-go values of shape (max_length, 1)
                - d (torch.Tensor): Done flags of shape (max_length, 1)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length, 1)
                - mask (torch.Tensor): Mask tensor of shape (max_length, 1)

        Note:
            - Observations are scaled by 255 during loading (as they are stored normalized)
            - Shorter trajectories are padded to max_length:
                - Observations are padded with the last state
                - Actions are padded with -10
                - Done flags are padded with 2
                - Masks are padded with 0
                - Return-to-go values are padded with 0
            - All tensors have an additional dimension of size 1 at the end
        """
        file_path = os.path.join(self.directory, self.filtered_list[idx])
        data = np.load(file_path)

        s = data['obs'] * 255.0 # because data is already divided by 255
        a = data['action']
        r = data['reward']
        d = data['done']

        length = s.shape[0]
        
        s = torch.from_numpy(s).float()
        if self.normalize == 1:
            s = s / 255.0

        
        s = s.unsqueeze(0)
        

        # if not ohe:
        a = torch.from_numpy(a).unsqueeze(0).unsqueeze(-1)
        # IF OHE: a = torch.from_numpy(a).unsqueeze(0)#.unsqueeze(-1)
        rtg = torch.from_numpy(self.discount_cumsum(r)).unsqueeze(0).unsqueeze(-1)
        d = torch.from_numpy(d).unsqueeze(0).unsqueeze(-1).to(dtype=torch.long)
       
        timesteps = torch.from_numpy(np.arange(0, self.max_length).reshape(1, -1, 1))
        mask = torch.ones_like(a)
        
        pad_length = self.max_length - length
        state_to_pad = s[:, -1:, :, :, :].expand(1, pad_length, 3, 84, 84)
        s = torch.cat((s, state_to_pad), dim=1)

        a = F.pad(a, (0, 0, 0, pad_length, 0, 0), value=-10)
        d = F.pad(d, (0, 0, 0, pad_length, 0, 0), value=2)
        mask = F.pad(mask, (0, 0, 0, pad_length, 0, 0), value=0)
        rtg = F.pad(rtg, (0, 0, 0, pad_length, 0, 0), value=0)

        return s.squeeze(), a.squeeze().unsqueeze(-1), rtg.squeeze().unsqueeze(-1), d.squeeze().unsqueeze(-1), timesteps.squeeze().unsqueeze(-1), mask.squeeze().unsqueeze(-1)


# Assuming 'directory_path' is the path to the directory containing .npz files
# dataset = ViZDoomIterDataset('../VizDoom_data/iterative_data/', gamma=1.0, max_length=90, normalize=1)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)