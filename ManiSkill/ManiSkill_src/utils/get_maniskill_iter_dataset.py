import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class ManiSkillIterDataset(Dataset):
    def __init__(self, directory, gamma, max_length, normalize):
        """_summary_

        Args:
            directory (str): path to the directory with data files
            gamma (float): discount factor
            max_length (int): maximum number of timesteps used in batch generation
                                (max in dataset: 1001)
        """
        
        self.global_ind = 0
        self.directory = directory
        self.file_list = os.listdir(directory)
        # print(self.file_list)
        self.gamma = gamma
        self.max_length = max_length
        self.normalize = normalize
        self.filtered_list = []
        print('Filtering data...')
        self.filter_trajectories()
        print("Filtering complete. Number of .h5 files: ", len(self.filtered_list))

        print('Decoupling .h5 files into single trajectories...')
        self.list_of_trajectories = []
        self.decouple_h5_files()
        print('Decoupling complete. Number of trajectories: ', len(self.list_of_trajectories))

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

    def filter_trajectories(self):
        """
        Select only .h5 files with trajectories (each file consists of multiple trajectories)
        """
        for idx in tqdm(range(len(self.file_list))):
            if self.file_list[idx].endswith('.h5'):
                self.filtered_list.append(self.file_list[idx])

    def decouple_h5_files(self):
        # Create a folder at the same level as self.directory if it doesn't exist
        parent_dir = os.path.dirname(os.path.dirname(self.directory))
        new_folder = os.path.join(parent_dir, os.path.basename(self.directory.rstrip('/')) + "_" + "single_h5/")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        for file_path in tqdm(self.filtered_list):
            with h5py.File(self.directory+file_path, "r") as f:
                for i in range(len(f.keys())):
                    traj = f[f"traj_{i}"]
                    a = traj["actions"][()]
                    o = traj["obs"]["rgb"][()]
                    d = np.logical_or(traj["terminated"][()], traj["truncated"][()])
                    r = traj["rewards"][()]
                    # print(file_path)
                    path = new_folder+"trajectory"+"_"+str(self.global_ind)+".npz"
                    self.list_of_trajectories.append(path)
                    np.savez(path, a=a, o=o, d=d, r=r)
                    self.global_ind += 1

            # if self.global_ind >= 128: # ! delete after debugging
            #     break

    def __len__(self):
        return len(self.list_of_trajectories)

    def __getitem__(self, idx):
        """
        obs: (T, H, W, C) -> (T, C, H, W)
        """
        file_path = os.path.join(self.list_of_trajectories[idx])
        data = np.load(file_path)

        s = data['o']
        a = data['a']
        r = data['r']
        d = data['d']

        s = torch.from_numpy(s).float().permute(0, 3, 1, 2)

        if self.normalize == 1:
            s = s / 255.0
        
        s = s.unsqueeze(0)

        a = torch.from_numpy(a).unsqueeze(0)
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

# dataset = ManiSkillIterDataset("./dataset_pushcube_v1/", gamma=1.0, max_length=50, normalize=1)
# batch_size = 128
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)