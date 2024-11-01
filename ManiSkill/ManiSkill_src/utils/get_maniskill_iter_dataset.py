import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

# try:
#     import multiprocessing
#     multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass

# TODO: add feature to just filter .npz files fast and to decouple them only once


class ManiSkillIterDataset(Dataset):
    def __init__(self, directory, gamma, max_length, normalize, h5_to_npz=False):
        """_summary_

        Args:
            directory (str): path to the directory with data files
            gamma (float): discount factor
            max_length (int): maximum number of timesteps used in batch generation
                                (max in dataset: 50)
        """
        
        self.global_ind = 0
        self.h5_to_npz = h5_to_npz
        self.directory = directory
        self.file_list = os.listdir(directory)
        # print(self.file_list)
        self.gamma = gamma
        self.max_length = max_length
        self.normalize = normalize

        if self.h5_to_npz:
            self.filtered_list_h5 = []
            print('Filtering .h5 data...')
            self.filter_trajectories_h5()
            print("Filtering complete. Number of .h5 files: ", len(self.filtered_list_h5))

            print('Decoupling .h5 files into single trajectories...')
            self.list_of_trajectories = []
            self.decouple_h5_files()
            print('Decoupling complete. Number of trajectories: ', len(self.list_of_trajectories))
        else:
            self.filtered_list_npz = []
            print('Filtering .npz data...')
            self.filter_trajectories_npz()
            print('Filtering complete. Number of trajectories: ', len(self.filtered_list_npz))


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

    def filter_trajectories_h5(self):
        """
        Select only .h5 files with trajectories (each file consists of multiple trajectories)
        """
        for idx in tqdm(range(len(self.file_list))):
            if self.file_list[idx].endswith('.h5'):
                self.filtered_list_h5.append(self.file_list[idx])
        
    def filter_trajectories_npz(self):
        """
        Select only .npz files with trajectories if directory contains .npz files exists (after decoupling of .h5 files)
        """
        parent_dir = os.path.dirname(os.path.dirname(self.directory))
        new_folder = os.path.join(parent_dir, os.path.basename(self.directory.rstrip('/')) + "_" + "single_h5/")
        self.new_folder = new_folder
        if os.path.exists(new_folder):
            for file_path in tqdm(os.listdir(new_folder)):
                if file_path.endswith('.npz'):
                    self.filtered_list_npz.append(file_path)
        else:
            raise FileNotFoundError(f"Directory {new_folder} does not exist")


    def decouple_h5_files(self):
        # Create a folder at the same level as self.directory if it doesn't exist
        parent_dir = os.path.dirname(os.path.dirname(self.directory))
        new_folder = os.path.join(parent_dir, os.path.basename(self.directory.rstrip('/')) + "_" + "single_h5/")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        for file_path in tqdm(self.filtered_list_h5):
            with h5py.File(self.directory+file_path, "r") as f:
                for i in range(len(f.keys())):
                    traj = f[f"traj_{i}"]
                    a = traj["actions"][()]
                    o = traj["obs"]["rgb"][()]
                    d = np.logical_or(traj["terminated"][()], traj["truncated"][()])
                    r = traj["rewards"][()]
                    # print(file_path)
                    path = new_folder + "trajectory" + "_" + str(self.global_ind) + ".npz"
                    self.list_of_trajectories.append(path)
                    np.savez(path, a=a, o=o, d=d, r=r)
                    self.global_ind += 1

                    del a, o, d, r

            # if self.global_ind >= 1024: # ! delete after debugging
            #     break

    def __len__(self):
        if self.h5_to_npz:
            return len(self.list_of_trajectories)
        else:
            return len(self.filtered_list_npz)

    def __getitem__(self, idx):
        """
        obs: (T, H, W, C) -> (T, C, H, W)
        """
        if self.h5_to_npz:
            file_path = os.path.join(self.list_of_trajectories[idx])
        else:
            file_path = os.path.join(self.new_folder, self.filtered_list_npz[idx])

        with np.load(file_path, allow_pickle=False) as data:
            s = data['o'].copy()
            a = data['a'].copy()
            r = data['r'].copy()
            d = data['d'].copy()

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