ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from offline_rl_baselines.diffusion_policy.diffusion_policy.conditional_unet1d import ConditionalUnet1D
from offline_rl_baselines.diffusion_policy.diffusion_policy.evaluate import evaluate
from offline_rl_baselines.diffusion_policy.diffusion_policy.make_env import make_eval_envs
from offline_rl_baselines.diffusion_policy.diffusion_policy.plain_conv import PlainConv
from offline_rl_baselines.diffusion_policy.diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)

from src.envs_datasets.mikasa_robo_dataset import MIKASARoboIterDataset
from src.envs.mikasa_robo.mikasa_robo_initialization import InitializeMikasaRoboEnv

# python3 offline_rl_baselines/diffusion_policy_raw_code/train_rgbd.py --env_id mikasa_robo_RememberColor3-v0 --total_iters 10_000 --unet_dims 128 256 512 --diffusion_step_embed_dim 128 --batch_size 128 --obs_horizon 2 --act_horizon 8 --pred_horizon 16 --n_groups 8 --max_episode_steps 60 --max_length 60 --track --demo_path "data/MIKASA-Robo/unbatched/RememberColor3-v0" --seed 1

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "DP-MIKASA-Robo"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "mikasa_robo_RememberColor3-v0"
    """the id of the environment"""
    demo_path: str = None
    """the path of demo dataset"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 32
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])  # default setting is about ~4.5M params
    n_groups: int = 8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value."""
    log_freq: int = 250
    """the frequency of logging the training metrics"""
    eval_freq: int = 250
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = 250
    """the frequency of saving the model checkpoints."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments."""
    num_dataload_workers: int = 6 # 0
    """the number of workers to use for loading the training data"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments."""

    # MIKASA-Robo specific arguments
    gamma: float = 1.0
    """discount factor for RTG calculation"""
    max_length: int = 90
    """maximum sequence length"""
    normalize: int = 1
    """whether to normalize observations"""


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        from diffusion_policy.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, obs_space
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(
                    _obs_traj_dict["depth"].astype(np.float32)
                ).to(device=device, dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(
                    device
                )  # still uint8
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(
                device
            )
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(
                device=device
            )
        print(
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in args.control_mode
            or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            print("Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            # NOTE for absolute joint pos control probably should pad with the final joint position action.
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            args.obs_horizon,
            args.pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        
        # For MIKASA-Robo, we know the action space shape
        self.act_dim = 8  # Assuming 7-DoF robot arm + gripper
        
        # Visual encoder for RGB+Depth images
        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=6,  # RGB (3) + Depth (3)
            out_dim=visual_feature_dim,
            pool_feature_map=True
        )
        
        # State dimension (if any additional state features)
        state_dim = 0  # Modify if you have additional state features
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_horizon * (visual_feature_dim + state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def encode_obs(self, obs_seq, eval_mode=False):
        # obs_seq shape: [B, obs_horizon, C, H, W]
        # Data is already normalized in dataset if normalize=1
        # Just convert to float
        obs_seq = obs_seq.float()
        
        batch_size = obs_seq.shape[0]
        # Flatten batch and horizon dimensions
        img_seq = obs_seq.flatten(end_dim=1)  # [B*obs_horizon, C, H, W]
        
        # Get visual features
        visual_feature = self.visual_encoder(img_seq)  # [B*obs_horizon, D]
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # [B, obs_horizon, D]

        feature = visual_feature
        
        return feature.flatten(start_dim=1)  # [B, obs_horizon * D]

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=obs_seq.device
        ).long()

        # add noise to the clean actions according to the noise magnitude
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq.device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # [B, act_horizon, act_dim]


def save_ckpt(run_name, tag, path_to_save):
    os.makedirs(f"{path_to_save}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
        },
        f"{path_to_save}/checkpoints/{tag}.pt",
    )


@torch.no_grad()
def run_inference(env, agent, num_episodes=2, device='cuda', max_episode_steps=90):
    agent.eval()
    
    # Reset environment and get initial state
    state_0, _ = env.reset()
    state_0 = state_0['rgb']  # envs_numx128x128x6
    
    # Convert state to expected format [C, H, W] and normalize if needed
    state = state_0.float().permute(0, 3, 1, 2).to(device)  # envs_numx6x128x128
    if args.normalize == 1:
        state = state / 255.0
    state = state.unsqueeze(1)  # envs_numx1x6x128x128
    
    # Initialize observation sequence
    obs_seq = state.repeat(1, agent.obs_horizon, 1, 1, 1)  # [envs_num, obs_horizon, C, H, W]

    eval_metrics = defaultdict(lambda: torch.zeros(0, device=device, dtype=torch.float32))
    
    # Initialize metrics tracking
    episode_return = torch.zeros((num_episodes), device=device, dtype=torch.float32)
    episode_length = torch.zeros((num_episodes), device=device, dtype=torch.float32)
    done = torch.zeros((num_episodes), dtype=torch.bool, device=device)
    
    completed_returns = []
    completed_lengths = []
    
    for t in range(max_episode_steps):
        # Get action from agent
        action = agent.get_action(obs_seq)  # [envs_num, act_horizon, act_dim]
        
        # Execute first action from the sequence
        state, reward, terminated, truncated, eval_infos = env.step(action[:, 0])
        
        # Update metrics
        reward = eval_infos['success'].float()  # Use success as reward
        episode_return += reward
        episode_length += 1
        
        # Update observation sequence
        state = state['rgb'].float().permute(0, 3, 1, 2).to(device)
        if args.normalize == 1:
            state = state / 255.0
        state = state.unsqueeze(1)
        obs_seq = torch.cat([obs_seq[:, 1:], state], dim=1)
        
        # Track metrics
        if "final_info" in eval_infos:
            for k, v in eval_infos["final_info"]["episode"].items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                else:
                    v = torch.tensor(v, device=device, dtype=torch.float32)
                eval_metrics[k] = torch.cat([eval_metrics[k], v])
        
        # Check if any episode is done
        done = torch.logical_or(terminated, truncated)
        if done.any():
            done_envs = done.nonzero().squeeze(-1)
            completed_returns.extend(episode_return[done_envs].tolist())
            completed_lengths.extend(episode_length[done_envs].tolist())
            # Сбросить только для новых эпизодов
            episode_return[done_envs] = 0
            episode_length[done_envs] = 0
            # Reset environment for done episodes
            if len(done_envs) > 0:
                reset_states, _ = env.reset()
                reset_states = reset_states['rgb'].float().permute(0, 3, 1, 2).to(device)
                reset_states = reset_states / 255.0
                reset_states = reset_states.unsqueeze(1)
                obs_seq[done_envs] = reset_states.repeat(1, agent.obs_horizon, 1, 1, 1)
    
    agent.train()
    
    # Calculate final metrics
    if len(completed_returns) > 0:
        mean_return = np.mean(completed_returns)
        mean_length = np.mean(completed_lengths)
    else:
        mean_return = episode_return.mean().item()
        mean_length = episode_length.mean().item()
    
    # Log additional metrics if available
    for k, v in eval_metrics.items():
        if len(v) > 0:
            writer.add_scalar(f"eval/{k}", v.mean().item(), iteration)
    
    return mean_return, mean_length


def sliding_window_batch(s, a, obs_horizon, pred_horizon):
    B, L, C, H, W = s.shape
    windows = []
    actions = []
    for b in range(B):
        for t in range(L - obs_horizon - pred_horizon + 2):
            obs_window = s[b, t:t+obs_horizon]
            act_window = a[b, t:t+pred_horizon]
            windows.append(obs_window)
            actions.append(act_window)
    obs_windows = torch.stack(windows)
    act_windows = torch.stack(actions)
    return obs_windows, act_windows


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # path to save
    path_to_save = f"runs/MIKASA_Robo/{args.env_id.split('_')[-1]}/diffusion_policy/{args.seed}/{run_name}"

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print('\n\n\nDevice: ', device, '\n\n\n')

    # Create dataset using your MIKASA-Robo dataset class
    dataset = MIKASARoboIterDataset(
        directory=args.demo_path,
        gamma=args.gamma,
        max_length=args.max_length,
        normalize=args.normalize
    )

    # Create dataloader
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    # Create environment using InitializeMikasaRoboEnv
    config = {
        "online_inference": {
            "episode_timeout": args.max_episode_steps if args.max_episode_steps is not None else 100
        }
    }
    envs = InitializeMikasaRoboEnv.create_mikasa_robo_env(
        env_name=args.env_id,
        run_dir=path_to_save,
        config=config
    )

    # Initialize agent
    agent = Agent(envs, args).to(device)
    optimizer = optim.AdamW(
        params=agent.parameters(),
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    if args.track:
        import wandb
        config = vars(args)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"],
        )
    writer = SummaryWriter(path_to_save)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    best_eval_return = float('-inf')  # ДО цикла обучения
    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # Training loop
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    
    for iteration, batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # Unpack batch from your dataset
        s, a, rtg, d, timesteps, masks = batch
        s = s.to(device)
        a = a.to(device)
        obs_windows, act_windows = sliding_window_batch(
            s, a, args.obs_horizon, args.pred_horizon
        )
        total_loss = agent.compute_loss(
            obs_seq=obs_windows,
            action_seq=act_windows,
        )
        timings["forward"] += time.time() - last_tick

        # Backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        timings["backward"] += time.time() - last_tick

        # EMA step
        last_tick = time.time()
        # if iteration % 4 == 0:
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        # Run inference every 10 iterations
        if iteration % args.eval_freq == 0:
            eval_return, eval_length = run_inference(envs, agent, num_episodes=20, device=device, max_episode_steps=args.max_episode_steps)
            writer.add_scalar("eval/return", eval_return, iteration)
            writer.add_scalar("eval/episode_length", eval_length, iteration)
            pbar.set_postfix({
                "loss": total_loss.item(),
                "eval_return": eval_return,
                "eval_length": eval_length
            })
            # Сохраняем лучший чекпоинт
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                # Сохраняем best_checkpoint.pth на уровень выше checkpoints
                best_ckpt_dir = os.path.join(path_to_save)
                best_ckpt_path = os.path.join(best_ckpt_dir, "best_checkpoint.pth")
                torch.save({
                    "agent": agent.state_dict(),
                    "ema_agent": ema_agent.state_dict(),
                    "iteration": iteration,
                    "eval_return": eval_return
                }, best_ckpt_path)

        # Logging
        writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
        pbar.set_postfix({
            "loss": total_loss.item(),
            "eval_return": eval_return,
            "eval_length": eval_length
        })
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            # writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

        if args.track:
            wandb.log({"iteration": iteration})

        # Save checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration), path_to_save)

        pbar.update(1)
        # if iteration % 10 != 0:  # Don't update postfix if we just did inference
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    writer.close()