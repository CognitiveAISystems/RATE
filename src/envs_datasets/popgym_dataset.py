import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
from src.utils.additional_data_processors import coords_to_idx, idx_to_coords


class POPGymDataset(Dataset):
    def __init__(self, directory, gamma, max_length, env_name):
        """_summary_

        Args:
            directory (str): path to the directory with data files
            gamma (float): discount factor
            max_length (int): maximum number of timesteps used in batch generation
                                (max in dataset: 1001)
            only_non_zero_rewards (bool): if True then use only trajectories
                                            with non-zero reward in the first
                                            max_length timesteps
        """
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.gamma = gamma
        self.max_length = max_length
        self.env_name = env_name

    def discount_cumsum(self, x):
        """
        Compute the discount cumulative sum of a 1D array.

        Args:
            x (ndarray): 1D array of values.

        Returns:
            ndarray: Discount cumulative sum of the input array.
        """
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
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

        if 'Battleship' in self.env_name:
            a = coords_to_idx(a.squeeze(-1), board_size=8)
        elif 'MineSweeper' in self.env_name:
            a = coords_to_idx(a.squeeze(-1), board_size=4)

        #Pad with zeros if length is less than max_length
        if length <= self.max_length:
            pad_length = self.max_length - length# + 1
            #print(s.shape)
            state_to_pad = s[:, -1, :].unsqueeze(1)# * 0
            while s.shape[1] < self.max_length:
                s = torch.cat((s, state_to_pad), dim=1)

            # action_to_pad = (a[:, -1, :].unsqueeze(1) * 0) - 10
            # while a.shape[1] < self.max_length:
            #     a = torch.cat((a, action_to_pad), dim=1)

            mask_to_pad = mask[:, -1, :].unsqueeze(1)# * 0
            while mask.shape[1] < self.max_length:
                mask = torch.cat((mask, mask_to_pad), dim=1)
            #print(a[:, -1].item())
            a = F.pad(a, (0, 0, 0, pad_length+1, 0, 0), value=-10)
            d = F.pad(d, (0, 0, 0, pad_length, 0, 0), value=2)
            timesteps = F.pad(timesteps, (0, 0, 0, pad_length, 0, 0), value=0)
            # mask = F.pad(mask, (0, 0, 0, pad_length+1, 0, 0), value=0)
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