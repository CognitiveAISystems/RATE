import numpy as np
from src.envs.tmaze.tmaze import TMazeClassicPassive
import glob
import pickle
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
import random


def TMaze_data_generator(max_segments: int, multiplier: int, hint_steps: int, 
                        desired_reward: float = 1.0, win_only: bool = True, 
                        segment_length: int = 30) -> None:
    """Generate and save TMaze trajectory data for different segment lengths.

    This function generates trajectory data for TMaze environments with varying corridor
    lengths and saves them as pickle files. It ensures data availability for training
    by generating missing data files.

    Args:
        max_segments: Maximum number of segments to generate data for.
        multiplier: Number of trajectories to generate for each segment length.
        hint_steps: Number of initial steps where the hint is preserved.
        desired_reward: Target reward value for successful trajectories.
        win_only: If True, only generate winning trajectories.
        segment_length: Base length for segment calculations.

    Note:
        Generated data is saved in 'data/TMaze/' directory with filenames
        following the pattern 'data_T_{segment_length}.pickle'.
    """
    current_directory = glob.os.getcwd()
    gen_flag= False
    for i in range(1, max_segments+1):
        # name = f'new_tmaze_data_segment_{i}_multiplier_{multiplier}_hint_steps_{hint_steps}_desired_reward_{desired_reward}_win_only_{win_only}_segment_length_{segment_length}'
        name = f'data_T_{i*segment_length}'
        data_path = current_directory + '/data/TMaze/'
        save_path = data_path + f"{name}.pickle"
        if not os.path.exists(save_path):
            if i == 1: 
                print("Data is not available. Generating...")
                gen_flag = True
            
            generate_dict_with_trajectories(segments=i, multiplier=multiplier, hint_steps=hint_steps, 
                                            desired_reward=desired_reward, win_only=win_only, segment_length=segment_length)
        else:
            if i == 1: print("Data is available.")
        if i == max_segments and gen_flag == True: print("Data successfully generated.")
            

class TMaze_dataset(Dataset):
    """A PyTorch Dataset for loading and processing TMaze environment trajectories.
    
    This dataset handles loading and preprocessing of TMaze environment trajectories
    stored as pickle files. It processes observations, actions, and rewards from
    the TMaze environment with support for different trajectory modes.

    Attributes:
        data (dict): Dictionary containing trajectory data loaded from pickle file.
        gamma (float): Discount factor for computing return-to-go (RTG).
        max_length (int): Maximum sequence length for trajectory segments.
        mode (str): Trajectory processing mode:
            - "equal": Fixed length trajectories (90/89/90)
            - "diff": Variable length trajectories (50/49/50 or 76/77/76) padded to max_length

    Note:
        The dataset expects each trajectory to contain:
        - 'obs': Observations of shape (T, channels) containing:
            - y position
            - hint
            - flag
            - noise
        - 'action': Discrete actions in [0, 1, 2, 3]
        - 'rtg': Return-to-go values
        - 'done': Episode termination flags
    """

    def __init__(self, path: str, gamma: float, mode: str, max_length: int):
        """Initialize the TMaze dataset.

        Args:
            path: Path to the pickle file containing trajectory data.
            gamma: Discount factor for computing return-to-go values.
            mode: Trajectory processing mode ("equal" or "diff").
            max_length: Maximum sequence length for trajectory segments.
        """
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.gamma = gamma
        self.max_length = max_length
        self.mode = mode
        """
        equal - 90/89/90
        diff - 50/49/50 or 76/77/76 padded to 90/89/90
        """
        
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
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum

    def __getitem__(self, index: int) -> tuple:
        """Get a trajectory segment by index.

        Args:
            index: Index of the trajectory to retrieve.

        Returns:
            A tuple containing:
                - s (torch.Tensor): Observations of shape (max_length, channels)
                - a (torch.Tensor): Actions of shape (max_length,)
                - rtg (torch.Tensor): Return-to-go values of shape (max_length,)
                - d (torch.Tensor): Done flags of shape (max_length,)
                - timesteps (torch.Tensor): Timestep indices of shape (max_length,)
                - mask (torch.Tensor): Mask tensor of shape (max_length,)

        Note:
            In "diff" mode, shorter trajectories are padded to max_length:
                - Observations are padded with the last state
                - Actions are padded with -10
                - Done flags are padded with 2
                - Masks are padded with 0
                - Return-to-go values are padded with 0
        """
        channels = self.data[index]['obs'].shape[1]
        # channels = 4
        if self.mode == "equal":
            traj = self.data[index]
            length = traj['rtg'].shape[0]
            s = traj['obs']
            s = torch.from_numpy(s).float()
            s = s.reshape(1, length, channels)
            a = torch.from_numpy(traj['action'])
            a = a.reshape(1, length-1, 1)
            d = torch.from_numpy(traj['done'].reshape(1, length-1, 1)).to(dtype=torch.long)
            timesteps = torch.from_numpy(np.arange(0, length).reshape(1, -1, 1))
            rtg = torch.from_numpy(traj['rtg']).reshape(1, length, 1)
            mask = torch.ones_like(a)
            
        elif self.mode == "diff":
            traj = self.data[index]
            length = traj['rtg'].shape[0]
            s = traj['obs']
            s = torch.from_numpy(s).float()
            s = s.reshape(1, length, channels)
            a = torch.from_numpy(traj['action'])
            a = a.reshape(1, length-1, 1)
            d = torch.from_numpy(traj['done'].reshape(1, length-1, 1)).to(dtype=torch.long)
            timesteps = torch.from_numpy(np.arange(0, length).reshape(1, -1, 1))
            rtg = torch.from_numpy(traj['rtg']).reshape(1, length, 1)
            mask = torch.ones_like(a)

            #Pad with zeros if length is less than max_length
            if length <= self.max_length:
                pad_length = self.max_length - length# + 1
                #print(s.shape)
                state_to_pad = s[:, -1, :].unsqueeze(1)# * 0
                while s.shape[1] < self.max_length:
                    s = torch.cat((s, state_to_pad), dim=1)
                #print(a[:, -1].item())
                a = F.pad(a, (0, 0, 0, pad_length+1, 0, 0), value=-10)
                d = F.pad(d, (0, 0, 0, pad_length, 0, 0), value=2)
                timesteps = F.pad(timesteps, (0, 0, 0, pad_length, 0, 0), value=0)
                mask = F.pad(mask, (0, 0, 0, pad_length+1, 0, 0), value=0)
                rtg = F.pad(rtg, (0, 0, 0, pad_length, 0, 0), value=0)
        
        return s.squeeze(0), a.squeeze(0), rtg.squeeze(0), d.squeeze(), timesteps.squeeze(), mask.squeeze()
    
    def __len__(self):
        return len(self.data)
    
    
class CutDataset(Dataset):
    """A dataset wrapper that returns every other trajectory from the source dataset.
    
    This class is used to reduce the dataset size by taking only every second
    trajectory from the source dataset. Useful for creating smaller validation sets
    or balancing dataset sizes.

    Attributes:
        dataset (Dataset): Source dataset to sample from.
        length (int): Length of the cut dataset (half of source dataset).
    """

    def __init__(self, dataset: Dataset):
        """Initialize the cut dataset.

        Args:
            dataset: Source dataset to sample from.
        """
        self.dataset = dataset
        self.length = len(dataset) // 2

    def __getitem__(self, index):
        new_index = index * 2
        return self.dataset[new_index]

    def __len__(self):
        return self.length
    
class TMazeCombinedDataLoader:
    """A data loader that combines TMaze trajectories of different lengths.
    
    This class creates a combined dataset from multiple TMaze datasets with
    different corridor lengths, allowing training on a mixture of trajectory
    lengths. It supports both sequential and mixed dataset creation modes.

    Attributes:
        n_init (int): Initial number of segments to start with.
        n_final (int): Final number of segments to include.
        multiplier (int): Number of trajectories per segment length.
        hint_steps (int): Number of initial steps where hint is preserved.
        batch_size (int): Batch size for the data loader.
        mode (str): Dataset combination mode.
        desired_reward (float): Target reward for successful trajectories.
        win_only (bool): Whether to include only winning trajectories.
        segment_length (int): Base length for segment calculations.
        dataset (Dataset): Combined dataset.
        dataloader (DataLoader): PyTorch DataLoader for the combined dataset.

    Note:
        The combined dataset can be created in two modes:
        1. Sequential (one_mixed_dataset=False): Adds datasets of increasing
           segment lengths sequentially, optionally cutting previous datasets
        2. Mixed (one_mixed_dataset=True): Combines all segment lengths into
           a single mixed dataset
    """

    def __init__(self, n_init: int, n_final: int, multiplier: int, hint_steps: int,
                 batch_size: int, mode: str, cut_dataset: bool, 
                 one_mixed_dataset: bool = False, desired_reward: float = 1.0,
                 win_only: bool = True, segment_length: int = 30):
        """Initialize the combined data loader.

        Args:
            n_init: Initial number of segments to start with.
            n_final: Final number of segments to include.
            multiplier: Number of trajectories per segment length.
            hint_steps: Number of initial steps where hint is preserved.
            batch_size: Batch size for the data loader.
            mode: Dataset combination mode.
            cut_dataset: Whether to cut previous datasets when adding new ones.
            one_mixed_dataset: Whether to create a mixed dataset instead of sequential.
            desired_reward: Target reward for successful trajectories.
            win_only: Whether to include only winning trajectories.
            segment_length: Base length for segment calculations.
        """
        self.n_init = n_init
        self.n_final = n_final
        self.multiplier = multiplier
        self.hint_steps = hint_steps
        self.batch_size = batch_size
        self.mode = mode
        self.desired_reward = desired_reward
        self.win_only = win_only
        self.cut_dataset = cut_dataset
        self.one_mixed_dataset = one_mixed_dataset
        self.segment_length = segment_length
        self.dataset = self._generate_combined_dataset()
        self.dataloader = self._generate_dataloader()
    
    def _get_dataloaders(self, path, gamma, mode, max_length):
        train_dataset = TMaze_dataset(path, gamma, mode, max_length)
        return train_dataset

    def _get_segment_dataloaders(self, N, multiplier, hint_steps, maxN, mode, desired_reward=1.0, win_only=True):
        # name = f'new_tmaze_data_segment_{N}_multiplier_{multiplier}_hint_steps_{hint_steps}_desired_reward_{desired_reward}_win_only_{win_only}_segment_length_{self.segment_length}'
        name = f'data_T_{self.n_final*self.segment_length}'
        data_path = f'data/TMaze/{name}.pickle'

        train_dataset = self._get_dataloaders(path=data_path, gamma=1.0, mode="diff", max_length=self.segment_length*maxN)

        return train_dataset

    def _generate_combined_dataset(self):
        if self.one_mixed_dataset == False:
            dataset = self._get_segment_dataloaders(N=self.n_init, multiplier=self.multiplier, hint_steps=self.hint_steps, 
                                                    maxN=self.n_final, mode=self.mode, desired_reward=self.desired_reward,
                                                    win_only=self.win_only)
            for N in range(self.n_init+1, self.n_final + 1):

                if self.cut_dataset:
                    dataset = CutDataset(dataset)

                dataset_new = self._get_segment_dataloaders(N=N, multiplier=self.multiplier, hint_steps=self.hint_steps, 
                                                            maxN=self.n_final, mode=self.mode, desired_reward=self.desired_reward,
                                                            win_only=self.win_only)
                dataset = ConcatDataset([dataset, dataset_new])
                
        elif self.one_mixed_dataset == True:
            dataset = self._get_segment_dataloaders(N=self.n_init, multiplier=self.multiplier, hint_steps=self.hint_steps, 
                                                    maxN=self.n_final, mode=self.mode, desired_reward=self.desired_reward,
                                                    win_only=self.win_only)
            for N in range(2, self.n_final + 1):
                dataset_new = self._get_segment_dataloaders(N=N, multiplier=self.multiplier, hint_steps=self.hint_steps, 
                                                            maxN=self.n_final, mode=self.mode, desired_reward=self.desired_reward,
                                                            win_only=self.win_only)
                dataset = ConcatDataset([dataset, dataset_new])

        return dataset

    def _generate_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
# # Example of usage:
# combined_dataloader = TMazeCombinedDataLoader(n_init=1, n_final=2, multiplier=10, hint_steps=1, batch_size=2, mode=None, cut_dataset=False) #modes: "" for full and "mini_" for mini #jlast for last el junction
# train_dataloader = combined_dataloader.dataloader
# dataset = combined_dataloader.dataset
# len(train_dataloader), len(dataset)




def generate_trajectory(episode_timeout: int, corridor_length: int, hint_steps: int,
                       win: bool, seed_env: int, seed_noise: int,
                       desired_reward: float = 1.0) -> tuple:
    """Generate a single TMaze trajectory.

    This function generates a trajectory in the TMaze environment following
    an optimal policy based on the hint and win condition.

    Args:
        episode_timeout: Maximum number of steps in the episode.
        corridor_length: Length of the corridor in the maze.
        hint_steps: Number of initial steps where hint is preserved.
        win: Whether to generate a winning trajectory.
        seed_env: Environment seed (0 for down, 1 for up).
        seed_noise: Seed for noise generation.
        desired_reward: Target reward for successful trajectories.

    Returns:
        A tuple containing:
            - states (ndarray): Observations of shape (T, 4) containing:
                - y position
                - hint
                - flag
                - noise
            - actions (ndarray): Discrete actions in [0, 1, 2, 3]
            - returns_to_go (ndarray): Return-to-go values
            - dones (ndarray): Episode termination flags

    Note:
        The optimal policy follows these rules:
        1. First step: Move right (action 0)
        2. At corridor end:
            - If win=True: Move according to hint
            - If win=False: Move opposite to hint
        3. Otherwise: Move right (action 0)
    """
    # Initialization:
    np.random.seed(seed_env)
    env = TMazeClassicPassive(episode_length=episode_timeout, corridor_length=corridor_length, penalty=0, goal_reward=desired_reward) # return {x, y, hint}
    obs = env.reset() # {x, y, hint}
    np.random.seed(seed_noise)
    obs = np.concatenate((obs, np.array([0]))) # {x, y, hint, flag}
    obs = np.concatenate((obs, np.array([np.random.randint(low=-1, high=1+1)]))) # {x, y, hint, flag, noise}
    done = False
    DESIRED_REWARD = desired_reward
    rtg = DESIRED_REWARD
    obss, acts, rtgs, dones, trajectories = [], [], [rtg], [], []
    
    # Agent's motion:
    for t in range(episode_timeout):
        #act = env.action_space.sample()
        trajectories.append(obs)
        obss.append(obs)
        
        # Optimal policy:
        if t == 0:
            act = 0
        elif obs[0] == corridor_length:
            if win == True:
                if trajectories[0][2] == -1:
                    act = 3 # down
                else:
                    act = 1 # up
            else:
                if trajectories[0][2] == -1:
                    act = 1
                else:
                    act = 3
        else:
            act = 0 # right
            
        obs, rew, done, info = env.step(act)
        rtg = rtg - (rew)
        """ HINT N STEPS """
        if t <= hint_steps - 2:
            obs[2] = trajectories[0][2]
        """ ============ """
        acts.append(act)
        rtgs.append(rtg)
        dones.append(int(done))
        
        # {x, y, hint} -> {x, y, hint, flag}
        if t != corridor_length-1:
            obs = np.concatenate((obs, np.array([0])))
        else:
            obs = np.concatenate((obs, np.array([1])))
        # {x, y, hint, flag} -> {x, y, hint, flag, noise}
        obs = np.concatenate((obs, np.array([np.random.randint(low=-1, high=1+1)])))
        
        if done:
            obss.append(obs)
            break

    return np.array(obss)[:, 1:], np.array(acts), np.array(rtgs), np.array(dones) #[:, 1:]


def generate_dict_with_trajectories(segments: int, multiplier: int, hint_steps: int,
                                  desired_reward: float = 1.0, win_only: bool = True,
                                  segment_length: int = 30) -> None:
    """Generate and save a dictionary of TMaze trajectories.

    This function generates multiple trajectories for the TMaze environment
    and saves them as a dictionary in a pickle file. It supports both
    win-only and mixed win/loss trajectory generation.

    Args:
        segments: Number of segments to generate trajectories for.
        multiplier: Number of trajectories to generate per segment.
        hint_steps: Number of initial steps where hint is preserved.
        desired_reward: Target reward for successful trajectories.
        win_only: Whether to generate only winning trajectories.
        segment_length: Base length for segment calculations.

    Note:
        Generated data is saved as a dictionary where each key is an index
        and the value is a dictionary containing:
        - 'obs': Observations
        - 'action': Actions
        - 'rtg': Return-to-go values
        - 'done': Episode termination flags

        If win_only is False, the multiplier is split evenly between
        winning and losing trajectories.
    """
    current_directory = glob.os.getcwd()
    # name = f'new_tmaze_data_segment_{segments}_multiplier_{multiplier}_hint_steps_{hint_steps}_desired_reward_{desired_reward}_win_only_{win_only}_segment_length_{segment_length}'
    name = f'data_T_{segments*segment_length}'
    data_path = current_directory + '/data/TMaze/'
    isExist = os.path.exists(data_path)
    save_path = data_path + f"{name}.pickle"
    if not isExist:
        os.makedirs(data_path)

    files = glob.glob(save_path + '*')
    for f in files:
        os.remove(f)

    data = {}
    iteration = 0
    if win_only:
        for seed_env in [0, 1]:
            for seed_noise in tqdm(range(multiplier)):
                L = random.randint(1, segment_length*segments-2)
                obss, acts, rtgs, dones = generate_trajectory(
                    episode_timeout=L+2, 
                    corridor_length=L, 
                    hint_steps=hint_steps, win=True, seed_env=seed_env, 
                    seed_noise=seed_noise, desired_reward=desired_reward
                )
                
                data[iteration] = {'obs': obss, 'action': acts, 'rtg': rtgs, 'done': dones}
                iteration += 1
    else:
        for seed_env in [0, 1]:
            for seed_noise in tqdm(range(multiplier//2)):
                L = random.randint(1, segment_length*segments-2)
                obss, acts, rtgs, dones = generate_trajectory(
                    episode_timeout=L+2, 
                    corridor_length=L, 
                    hint_steps=hint_steps, win=True, seed_env=seed_env, 
                    seed_noise=seed_noise, desired_reward=desired_reward
                )

                data[iteration] = {'obs': obss, 'action': acts, 'rtg': rtgs, 'done': dones}
                iteration += 1
            for seed_noise in tqdm(range(multiplier//2, multiplier)):
                L = random.randint(1, segment_length*segments-2)
                obss, acts, rtgs, dones = generate_trajectory(
                    episode_timeout=L+2, 
                    corridor_length=L, 
                    hint_steps=hint_steps, win=False, seed_env=seed_env, 
                    seed_noise=seed_noise, desired_reward=desired_reward
                )

                data[iteration] = {'obs': obss, 'action': acts, 'rtg': rtgs, 'done': dones}
                iteration += 1

    with open(save_path,'wb') as f:
            pickle.dump(data, f)