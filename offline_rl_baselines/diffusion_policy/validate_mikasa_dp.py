import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from tqdm import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv
from train_rgbd import Agent, Args
from src.envs.mikasa_robo.mikasa_robo_initialization import InitializeMikasaRoboEnv


@dataclass
class ValidateArgs:
    """Arguments for validation script"""
    ckpt_path: str = "runs/MIKASA_Robo/RememberColor3-v0/diffusion_policy/1/mikasa_robo_RememberColor3-v0__train_rgbd__1__1747077782/best_checkpoint.pth"
    """Path to the checkpoint file"""
    num_episodes: int = 100
    """Number of episodes to evaluate"""
    num_envs: int = 10
    """Number of parallel environments"""
    seed: int = 1
    """Random seed"""
    cuda: bool = True
    """Whether to use CUDA"""
    torch_deterministic: bool = True
    """Whether to use deterministic torch operations"""
    use_ema: bool = False
    """Whether to use EMA model instead of regular model"""
    save_videos: bool = False
    """Whether to save evaluation videos"""
    video_dir: Optional[str] = None
    """Directory to save videos (if save_videos is True)"""
    
    # Model parameters (should match training)
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    
    # Environment parameters
    env_id: str = "mikasa_robo_RememberColor3-v0"
    max_episode_steps: int = 60
    sim_backend: str = "physx_cpu"
    normalize: int = 1


def evaluate_mikasa(n: int, agent, eval_envs, device, sim_backend: str, args, progress_bar: bool = True):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    
    eval_metrics = defaultdict(list)
    completed_episodes = 0
    
    while completed_episodes < n:
        # Reset environment and get initial state
        state_0, _ = eval_envs.reset()
        state_0 = state_0['rgb']  # envs_numx128x128x6
        
        # Convert state to expected format [C, H, W] and normalize if needed
        state = state_0.float().permute(0, 3, 1, 2).to(device)  # envs_numx6x128x128
        if args.normalize == 1:
            state = state / 255.0
        state = state.unsqueeze(1)  # envs_numx1x6x128x128
        
        # Initialize observation sequence
        obs_seq = state.repeat(1, agent.obs_horizon, 1, 1, 1)  # [envs_num, obs_horizon, C, H, W]
        
        # Initialize metrics tracking
        episode_return = torch.zeros((eval_envs.num_envs), device=device, dtype=torch.float32)
        episode_length = torch.zeros((eval_envs.num_envs), device=device, dtype=torch.float32)
        done = torch.zeros((eval_envs.num_envs), dtype=torch.bool, device=device)
        
        for t in range(args.max_episode_steps):
            # Get action from agent
            action = agent.get_action(obs_seq)  # [envs_num, act_horizon, act_dim]
            
            # Execute first action from the sequence
            state, reward, terminated, truncated, eval_infos = eval_envs.step(action[:, 0])
            
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
                    eval_metrics[k].extend(v.cpu().numpy())
            
            # Check if any episode is done
            done = torch.logical_or(terminated, truncated)
            if done.any():
                done_envs = done.nonzero().squeeze(-1)
                # Record metrics for completed episodes
                for k, v in eval_metrics.items():
                    if len(v) > 0:
                        eval_metrics[k] = v[:-len(done_envs)]  # Remove metrics for current episodes
                eval_metrics['return'].extend(episode_return[done_envs].cpu().numpy())
                eval_metrics['length'].extend(episode_length[done_envs].cpu().numpy())
                
                completed_episodes += len(done_envs)
                if progress_bar:
                    pbar.update(len(done_envs))
                
                # Reset metrics for completed episodes
                episode_return[done_envs] = 0
                episode_length[done_envs] = 0
                
                if completed_episodes >= n:
                    break
    
    agent.train()
    if progress_bar:
        pbar.close()
    
    # Convert lists to numpy arrays
    for k in eval_metrics:
        eval_metrics[k] = np.array(eval_metrics[k])
    
    return eval_metrics

def main():
    # Parse arguments
    args = tyro.cli(ValidateArgs)
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Create training args for model initialization
    train_args = Args(
        env_id=args.env_id,
        seed=args.seed,
        cuda=args.cuda,
        obs_horizon=args.obs_horizon,
        act_horizon=args.act_horizon,
        pred_horizon=args.pred_horizon,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        unet_dims=args.unet_dims,
        n_groups=args.n_groups,
        max_episode_steps=args.max_episode_steps,
        num_eval_envs=args.num_envs,
        sim_backend=args.sim_backend,
        normalize=args.normalize
    )
    
    # Create environment
    print("Creating environment...")
    config = {
        "online_inference": {
            "episode_timeout": args.max_episode_steps
        }
    }
    envs = InitializeMikasaRoboEnv.create_mikasa_robo_env(
        env_name=args.env_id,
        run_dir=".",
        config=config
    )
    
    # Initialize agent
    print("Initializing agent...")
    agent = Agent(envs, train_args).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    
    # Load either EMA or regular model
    if args.use_ema and "ema_agent" in ckpt:
        print("Loading EMA model...")
        agent.load_state_dict(ckpt["ema_agent"])
    else:
        print("Loading regular model...")
        agent.load_state_dict(ckpt["agent"])
    
    # Print checkpoint info if available
    if "iteration" in ckpt:
        print(f"Checkpoint iteration: {ckpt['iteration']}")
    if "eval_return" in ckpt:
        print(f"Checkpoint eval return: {ckpt['eval_return']:.3f}")
    
    # Create video directory if needed
    if args.save_videos:
        if args.video_dir is None:
            args.video_dir = os.path.join(os.path.dirname(args.ckpt_path), "eval_videos")
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"Saving videos to {args.video_dir}")
    
    # Run evaluation
    print(f"\nRunning evaluation for {args.num_episodes} episodes...")
    eval_metrics = evaluate_mikasa(
        n=args.num_episodes,
        agent=agent,
        eval_envs=envs,
        device=device,
        sim_backend=args.sim_backend,
        args=args,
        progress_bar=True
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for k, v in eval_metrics.items():
        if len(v) > 0:
            mean = np.mean(v)
            std = np.std(v)
            min_v = np.min(v)
            max_v = np.max(v)
            print(f"{k}:")
            print(f"  Mean: {mean:.3f}")
            print(f"  Std:  {std:.3f}")
            print(f"  Min:  {min_v:.3f}")
            print(f"  Max:  {max_v:.3f}")
        else:
            print(f"{k}:")
            print("  No data collected for this metric.")
        print("-" * 50)
    
    # Save results to file
    results_dir = os.path.join(os.path.dirname(args.ckpt_path), "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(results_dir, f"eval_results_{timestamp}.txt")
    
    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        f.write(f"Number of episodes: {args.num_episodes}\n")
        f.write(f"Using EMA model: {args.use_ema}\n")
        f.write("=" * 50 + "\n\n")
        
        for k, v in eval_metrics.items():
            if len(v) > 0:
                f.write(f"{k}:\n")
                f.write(f"  Mean: {np.mean(v):.3f}\n")
                f.write(f"  Std:  {np.std(v):.3f}\n")
                f.write(f"  Min:  {np.min(v):.3f}\n")
                f.write(f"  Max:  {np.max(v):.3f}\n")
            else:
                f.write(f"{k}:\n")
                f.write("  No data collected for this metric.\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()


"""
# Base validation
python offline_rl_baselines/diffusion_policy_raw_code/validate_mikasa_dp.py

# Validation with other parameters
python offline_rl_baselines/diffusion_policy_raw_code/validate_mikasa_dp.py --num_episodes 200 --use_ema --save_videos

# Validation with other checkpoint
python offline_rl_baselines/diffusion_policy_raw_code/validate_mikasa_dp.py --ckpt_path path/to/your/checkpoint.pth

"""