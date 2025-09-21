"""
Online RL Training Script for MDP Environments

This script trains state-of-the-art online RL policies on classic control MDP tasks:
- CartPole-v1
- MountainCar-v0  
- MountainCarContinuous-v0
- Acrobot-v1
- Pendulum-v1

Uses Stable-Baselines3 with optimized hyperparameters for each environment.
"""

import gymnasium as gym
import numpy as np
import torch
import os
import json
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import argparse
import time
from tqdm import tqdm

from stable_baselines3 import PPO, SAC, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy


@dataclass
class TrainingConfig:
    """Configuration for training online RL policies."""
    env_name: str
    algorithm: str
    total_timesteps: int
    n_envs: int = 4
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    seed: int = 42
    device: str = "auto"
    save_freq: int = 50000
    verbose: int = 1


class MDPTrainer:
    """Trainer class for MDP environments using Stable-Baselines3."""
    
    # Optimized hyperparameters for each environment and algorithm
    HYPERPARAMETERS = {
        "CartPole-v1": {
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 32,
                "batch_size": 256,
                "n_epochs": 20,
                "gamma": 0.98,
                "gae_lambda": 0.8,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "total_timesteps": 100000
            },
            "DQN": {
                "learning_rate": 1e-3,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 128,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 100,
                "exploration_fraction": 0.16,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.04,
                "total_timesteps": 100000
            }
        },
        "MountainCar-v0": {
            "DQN": {
                "learning_rate": 4e-3,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 128,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 16,
                "gradient_steps": 8,
                "target_update_interval": 600,
                "exploration_fraction": 0.2,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.07,
                "total_timesteps": 120000
            },
            "PPO": {
                "learning_rate": 1e-3,
                "n_steps": 16,
                "batch_size": 256,
                "n_epochs": 4,
                "gamma": 0.99,
                "gae_lambda": 0.98,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "total_timesteps": 120000
            }
        },
        "MountainCarContinuous-v0": {
            "SAC": {
                "learning_rate": 3e-4,
                "buffer_size": 300000,
                "learning_starts": 10000,
                "batch_size": 512,
                "tau": 0.02,
                "gamma": 0.98,
                "train_freq": 64,
                "gradient_steps": 64,
                "ent_coef": "auto",
                "target_update_interval": 1,
                "total_timesteps": 300000
            },
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 512,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.9999,
                "gae_lambda": 0.9,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "total_timesteps": 300000
            }
        },
        "Acrobot-v1": {
            "DQN": {
                "learning_rate": 6.3e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 128,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 250,
                "exploration_fraction": 0.16,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.04,
                "total_timesteps": 100000
            },
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 256,
                "batch_size": 64,
                "n_epochs": 4,
                "gamma": 0.99,
                "gae_lambda": 0.94,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "total_timesteps": 100000
            }
        },
        "Pendulum-v1": {
            "SAC": {
                "learning_rate": 3e-4,
                "buffer_size": 300000,
                "learning_starts": 10000,
                "batch_size": 512,
                "tau": 0.02,
                "gamma": 0.98,
                "train_freq": 64,
                "gradient_steps": 64,
                "ent_coef": "auto",
                "target_update_interval": 1,
                "total_timesteps": 300000
            },
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 512,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.9,
                "gae_lambda": 0.9,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "total_timesteps": 300000
            }
        }
    }
    
    # Best algorithm for each environment (based on performance benchmarks)
    BEST_ALGORITHMS = {
        "CartPole-v1": "PPO",
        "MountainCar-v0": "DQN", 
        "MountainCarContinuous-v0": "SAC",
        "Acrobot-v1": "DQN",
        "Pendulum-v1": "SAC"
    }
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.save_dir = Path(f"src/additional/gen_mdp_data/models/{config.env_name}/{config.algorithm}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        set_random_seed(config.seed)
        
    def create_env(self, n_envs: int = 1, monitor: bool = True):
        """Create vectorized environment."""
        if monitor:
            def make_env():
                env = gym.make(self.config.env_name)
                env = Monitor(env)
                return env
        else:
            def make_env():
                return gym.make(self.config.env_name)
        
        if n_envs == 1:
            return DummyVecEnv([make_env])
        else:
            return SubprocVecEnv([make_env for _ in range(n_envs)])
    
    def get_algorithm_class(self, algorithm: str):
        """Get the algorithm class from string name."""
        algorithm_map = {
            "PPO": PPO,
            "SAC": SAC,
            "DQN": DQN,
            "A2C": A2C
        }
        return algorithm_map[algorithm]
    
    def create_model(self, env):
        """Create the RL model with optimized hyperparameters."""
        AlgorithmClass = self.get_algorithm_class(self.config.algorithm)
        
        # Get hyperparameters for this environment and algorithm
        if (self.config.env_name in self.HYPERPARAMETERS and 
            self.config.algorithm in self.HYPERPARAMETERS[self.config.env_name]):
            hyperparams = self.HYPERPARAMETERS[self.config.env_name][self.config.algorithm].copy()
            # Remove total_timesteps from hyperparams as it's not a model parameter
            hyperparams.pop("total_timesteps", None)
        else:
            hyperparams = {}
        
        # Determine policy type based on action space
        if hasattr(env, 'action_space'):
            action_space = env.action_space
        else:
            # For vectorized environments
            action_space = env.envs[0].action_space
        
        if isinstance(action_space, gym.spaces.Discrete):
            policy = "MlpPolicy"
        elif isinstance(action_space, gym.spaces.Box):
            policy = "MlpPolicy"
        else:
            policy = "MlpPolicy"  # Default fallback
        
        # Add required parameters
        hyperparams.update({
            "policy": policy,
            "env": env,
            "verbose": self.config.verbose,
            "device": self.config.device,
            "seed": self.config.seed
        })
        
        return AlgorithmClass(**hyperparams)
    
    def train(self):
        """Train the RL model."""
        print(f"Training {self.config.algorithm} on {self.config.env_name}")
        print(f"Device: {self.config.device}")
        print(f"Save directory: {self.save_dir}")
        
        # Create environments
        train_env = self.create_env(n_envs=self.config.n_envs)
        eval_env = self.create_env(n_envs=1)
        
        # Create model
        model = self.create_model(train_env)
        
        # Create callbacks
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=str(self.save_dir / "best_model"),
            log_path=str(self.save_dir / "logs"),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(self.save_dir / "checkpoints"),
            name_prefix=f"{self.config.algorithm}_{self.config.env_name}"
        )
        
        callbacks = [eval_callback, checkpoint_callback]
        
        # Get total timesteps from hyperparameters or use config default
        if (self.config.env_name in self.HYPERPARAMETERS and 
            self.config.algorithm in self.HYPERPARAMETERS[self.config.env_name]):
            total_timesteps = self.HYPERPARAMETERS[self.config.env_name][self.config.algorithm]["total_timesteps"]
        else:
            total_timesteps = self.config.total_timesteps
        
        print(f"Training for {total_timesteps} timesteps...")
        
        # Train the model
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        training_time = time.time() - start_time
        
        # Save final model
        model.save(str(self.save_dir / "final_model"))
        
        # Evaluate final model
        print("Evaluating final model...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=100, deterministic=True
        )
        
        # Save training results
        results = {
            "env_name": self.config.env_name,
            "algorithm": self.config.algorithm,
            "total_timesteps": total_timesteps,
            "training_time": training_time,
            "final_mean_reward": float(mean_reward),
            "final_std_reward": float(std_reward),
            "seed": self.config.seed,
            "hyperparameters": self.HYPERPARAMETERS.get(self.config.env_name, {}).get(self.config.algorithm, {})
        }
        
        with open(self.save_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Results saved to: {self.save_dir}")
        
        train_env.close()
        eval_env.close()
        
        return model, results


def main():
    parser = argparse.ArgumentParser(description="Train online RL policies on MDP environments")
    parser.add_argument("--env", type=str, required=True,
                       choices=["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", 
                               "Acrobot-v1", "Pendulum-v1", "all"],
                       help="Environment to train on")
    parser.add_argument("--algorithm", type=str, default="best",
                       choices=["PPO", "SAC", "DQN", "A2C", "best"],
                       help="RL algorithm to use ('best' uses the optimal algorithm for each env)")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Total timesteps to train (uses optimized defaults if not specified)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # List of environments to train
    if args.env == "all":
        envs_to_train = ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", 
                        "Acrobot-v1", "Pendulum-v1"]
    else:
        envs_to_train = [args.env]
    
    # Train on each environment
    all_results = []
    for env_name in envs_to_train:
        # Determine algorithm to use
        if args.algorithm == "best":
            algorithm = MDPTrainer.BEST_ALGORITHMS[env_name]
        else:
            algorithm = args.algorithm
        
        # Determine timesteps
        if args.timesteps is None:
            if (env_name in MDPTrainer.HYPERPARAMETERS and 
                algorithm in MDPTrainer.HYPERPARAMETERS[env_name]):
                timesteps = MDPTrainer.HYPERPARAMETERS[env_name][algorithm]["total_timesteps"]
            else:
                timesteps = 100000  # Default
        else:
            timesteps = args.timesteps
        
        # Create config
        config = TrainingConfig(
            env_name=env_name,
            algorithm=algorithm,
            total_timesteps=timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            device=args.device
        )
        
        # Train
        trainer = MDPTrainer(config)
        model, results = trainer.train()
        all_results.append(results)
        
        print(f"\nCompleted training for {env_name} with {algorithm}")
        print("=" * 50)
    
    # Save summary results
    summary_path = Path("src/additional/gen_mdp_data/training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll training completed! Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
