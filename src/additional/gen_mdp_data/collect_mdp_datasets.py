"""
MDP Dataset Collection Script

This script collects offline datasets from trained RL policies on classic control MDP tasks.
The datasets are saved in a format compatible with the existing RATE/MATL training pipeline.
"""

import gymnasium as gym
import numpy as np
import torch
import os
import json
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import argparse
from tqdm import tqdm

from stable_baselines3 import PPO, SAC, DQN, A2C
from stable_baselines3.common.utils import set_random_seed


@dataclass 
class CollectionConfig:
    """Configuration for dataset collection."""
    env_name: str
    model_path: str
    algorithm: str
    n_episodes: int = 1000
    max_episode_length: int = 1000
    seed: int = 42
    deterministic: bool = True
    save_dir: Optional[str] = None


class MDPDatasetCollector:
    """Collector class for MDP environment datasets."""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        
        # Set up save directory
        if config.save_dir is None:
            self.save_dir = Path(f"data/MDP/{config.env_name}")
        else:
            self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        set_random_seed(config.seed)
        
        # Load model
        self.model = self._load_model()
        
        # Create environment
        self.env = gym.make(config.env_name)
        
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        
    def _load_model(self):
        """Load the trained RL model."""
        algorithm_map = {
            "PPO": PPO,
            "SAC": SAC, 
            "DQN": DQN,
            "A2C": A2C
        }
        
        AlgorithmClass = algorithm_map[self.config.algorithm]
        model = AlgorithmClass.load(self.config.model_path)
        return model
        
    def _is_discrete_action_space(self):
        """Check if the action space is discrete."""
        return isinstance(self.env.action_space, gym.spaces.Discrete)
        
    def discount_cumsum(self, x: np.ndarray, gamma: float = 0.99) -> np.ndarray:
        """Compute the discounted cumulative sum of rewards."""
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum
        
    def collect_episode(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """Collect a single episode of data."""
        obs_list, action_list, reward_list, done_list = [], [], [], []
        
        obs, info = self.env.reset(seed=self.config.seed + episode_idx)
        done = False
        step = 0
        
        while not done and step < self.config.max_episode_length:
            obs_list.append(obs.copy())
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=self.config.deterministic)
            
            # Handle different action space types
            if self._is_discrete_action_space():
                action_to_store = int(action)
            else:
                action_to_store = action.copy() if isinstance(action, np.ndarray) else action
            
            action_list.append(action_to_store)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            reward_list.append(float(reward))
            done_list.append(int(done))
            
            step += 1
        
        # Convert to numpy arrays
        obs_array = np.array(obs_list)
        action_array = np.array(action_list)
        reward_array = np.array(reward_list)
        done_array = np.array(done_list)
        
        return {
            'obs': obs_array,
            'action': action_array, 
            'reward': reward_array,
            'done': done_array
        }
        
    def collect_dataset(self) -> Dict[str, Any]:
        """Collect the full dataset."""
        print(f"Collecting {self.config.n_episodes} episodes for {self.config.env_name}")
        
        all_episodes = []
        episode_returns = []
        episode_lengths = []
        
        for episode_idx in tqdm(range(self.config.n_episodes)):
            episode_data = self.collect_episode(episode_idx)
            
            # Save individual episode
            episode_path = self.save_dir / f"train_data_{episode_idx}.npz"
            np.savez(episode_path, **episode_data)
            
            # Track statistics
            episode_return = np.sum(episode_data['reward'])
            episode_length = len(episode_data['reward'])
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            all_episodes.append(episode_data)
        
        # Compute dataset statistics
        stats = self._compute_statistics(all_episodes, episode_returns, episode_lengths)
        
        # Save dataset metadata
        metadata = {
            'env_name': self.config.env_name,
            'algorithm': self.config.algorithm,
            'model_path': str(self.config.model_path),
            'n_episodes': self.config.n_episodes,
            'max_episode_length': self.config.max_episode_length,
            'seed': self.config.seed,
            'deterministic': self.config.deterministic,
            'observation_space': str(self.env.observation_space),
            'action_space': str(self.env.action_space),
            'statistics': stats
        }
        
        with open(self.save_dir / "dataset_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Dataset saved to: {self.save_dir}")
        print(f"Dataset statistics:")
        print(f"  Episodes: {stats['n_episodes']}")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Mean episode return: {stats['mean_return']:.3f} ± {stats['std_return']:.3f}")
        print(f"  Mean episode length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
        print(f"  Return range: [{stats['min_return']:.3f}, {stats['max_return']:.3f}]")
        
        return metadata
        
    def _compute_statistics(self, episodes: List[Dict], returns: List[float], lengths: List[int]) -> Dict[str, Any]:
        """Compute dataset statistics."""
        # Basic statistics
        returns_array = np.array(returns)
        lengths_array = np.array(lengths)
        
        stats = {
            'n_episodes': len(episodes),
            'total_steps': int(np.sum(lengths_array)),
            'mean_return': float(np.mean(returns_array)),
            'std_return': float(np.std(returns_array)),
            'min_return': float(np.min(returns_array)),
            'max_return': float(np.max(returns_array)),
            'median_return': float(np.median(returns_array)),
            'mean_length': float(np.mean(lengths_array)),
            'std_length': float(np.std(lengths_array)),
            'min_length': int(np.min(lengths_array)),
            'max_length': int(np.max(lengths_array)),
            'median_length': float(np.median(lengths_array))
        }
        
        # Observation space statistics
        all_obs = np.concatenate([ep['obs'] for ep in episodes], axis=0)
        stats['obs_shape'] = list(all_obs.shape[1:])
        stats['obs_mean'] = all_obs.mean(axis=0).tolist()
        stats['obs_std'] = all_obs.std(axis=0).tolist()
        stats['obs_min'] = all_obs.min(axis=0).tolist()
        stats['obs_max'] = all_obs.max(axis=0).tolist()
        
        # Action space statistics  
        all_actions = np.concatenate([ep['action'].reshape(-1) for ep in episodes], axis=0)
        if self._is_discrete_action_space():
            # Discrete actions
            unique_actions, counts = np.unique(all_actions, return_counts=True)
            stats['action_distribution'] = {int(action): int(count) for action, count in zip(unique_actions, counts)}
            stats['n_unique_actions'] = len(unique_actions)
        else:
            # Continuous actions
            if len(all_actions.shape) > 1:
                stats['action_shape'] = list(all_actions.shape[1:])
                stats['action_mean'] = all_actions.mean(axis=0).tolist()
                stats['action_std'] = all_actions.std(axis=0).tolist()
                stats['action_min'] = all_actions.min(axis=0).tolist()
                stats['action_max'] = all_actions.max(axis=0).tolist()
            else:
                stats['action_mean'] = float(all_actions.mean())
                stats['action_std'] = float(all_actions.std())
                stats['action_min'] = float(all_actions.min())
                stats['action_max'] = float(all_actions.max())
        
        # Reward statistics
        all_rewards = np.concatenate([ep['reward'] for ep in episodes], axis=0)
        stats['reward_mean'] = float(all_rewards.mean())
        stats['reward_std'] = float(all_rewards.std())
        stats['reward_min'] = float(all_rewards.min())
        stats['reward_max'] = float(all_rewards.max())
        stats['reward_median'] = float(np.median(all_rewards))
        
        return stats


def find_best_model(env_name: str, base_dir: str = "src/additional/gen_mdp_data/models") -> Tuple[str, str]:
    """Find the best trained model for an environment."""
    env_dir = Path(base_dir) / env_name
    
    if not env_dir.exists():
        raise FileNotFoundError(f"No models found for {env_name} in {env_dir}")
    
    best_model_path = None
    best_algorithm = None
    best_reward = float('-inf')
    
    # Look through all algorithm directories
    for algo_dir in env_dir.iterdir():
        if not algo_dir.is_dir():
            continue
            
        algorithm = algo_dir.name
        
        # Check for training results
        results_path = algo_dir / "training_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            
            mean_reward = results.get('final_mean_reward', float('-inf'))
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_algorithm = algorithm
                
                # Look for best model
                best_model_file = algo_dir / "best_model" / "best_model.zip"
                if best_model_file.exists():
                    best_model_path = str(best_model_file)
                else:
                    # Fallback to final model
                    final_model_file = algo_dir / "final_model.zip"
                    if final_model_file.exists():
                        best_model_path = str(final_model_file)
    
    if best_model_path is None:
        raise FileNotFoundError(f"No valid model found for {env_name}")
        
    return best_model_path, best_algorithm


def main():
    parser = argparse.ArgumentParser(description="Collect MDP datasets from trained RL policies")
    parser.add_argument("--env", type=str, required=True,
                       choices=["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", 
                               "Acrobot-v1", "Pendulum-v1", "all"],
                       help="Environment to collect data from")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model (auto-detects best model if not specified)")
    parser.add_argument("--algorithm", type=str, default=None,
                       help="Algorithm used (auto-detects if not specified)")
    parser.add_argument("--n-episodes", type=int, default=1000,
                       help="Number of episodes to collect")
    parser.add_argument("--max-episode-length", type=int, default=1000,
                       help="Maximum episode length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save dataset (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # List of environments to collect data from
    if args.env == "all":
        envs_to_collect = ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", 
                          "Acrobot-v1", "Pendulum-v1"]
    else:
        envs_to_collect = [args.env]
    
    # Collect data for each environment
    all_metadata = []
    for env_name in envs_to_collect:
        print(f"\nCollecting data for {env_name}")
        print("=" * 50)
        
        # Auto-detect best model if not specified
        if args.model_path is None:
            try:
                model_path, algorithm = find_best_model(env_name)
                print(f"Auto-detected best model: {model_path}")
                print(f"Algorithm: {algorithm}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print(f"Please train a model for {env_name} first or specify --model-path")
                continue
        else:
            model_path = args.model_path
            algorithm = args.algorithm
            if algorithm is None:
                raise ValueError("Must specify --algorithm when using --model-path")
        
        # Create collection config
        config = CollectionConfig(
            env_name=env_name,
            model_path=model_path,
            algorithm=algorithm,
            n_episodes=args.n_episodes,
            max_episode_length=args.max_episode_length,
            seed=args.seed,
            deterministic=args.deterministic,
            save_dir=args.save_dir
        )
        
        # Collect dataset
        collector = MDPDatasetCollector(config)
        metadata = collector.collect_dataset()
        all_metadata.append(metadata)
        
        print(f"Completed data collection for {env_name}")
    
    # Save summary metadata
    summary_path = Path("src/additional/gen_mdp_data/dataset_collection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nAll data collection completed! Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
