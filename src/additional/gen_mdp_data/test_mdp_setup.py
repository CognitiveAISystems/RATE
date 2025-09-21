"""
Test script to verify MDP setup is working correctly.
This script tests the basic functionality without running full training.
"""

import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.envs_datasets.mdp_dataset import MDPDataset, create_mdp_dataloader


def test_environments():
    """Test that all MDP environments can be created."""
    print("Testing environment creation...")
    
    environments = [
        "CartPole-v1",
        "MountainCar-v0", 
        "MountainCarContinuous-v0",
        "Acrobot-v1",
        "Pendulum-v1"
    ]
    
    for env_name in environments:
        try:
            env = gym.make(env_name)
            obs, info = env.reset(seed=42)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            print(f"  ✅ {env_name}: OK")
        except Exception as e:
            print(f"  ❌ {env_name}: {e}")
            return False
    
    return True


def test_dataset_creation():
    """Test MDPDataset creation with dummy data."""
    print("\nTesting dataset creation...")
    
    # Create temporary directory with dummy data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy episode data
        n_episodes = 5
        for i in range(n_episodes):
            episode_length = np.random.randint(10, 50)
            
            # Dummy CartPole data
            obs = np.random.randn(episode_length, 4)  # 4D state space
            actions = np.random.randint(0, 2, size=episode_length)  # Binary actions
            rewards = np.random.randn(episode_length)
            dones = np.zeros(episode_length)
            dones[-1] = 1  # Last step is done
            
            data = {
                'obs': obs,
                'action': actions,
                'reward': rewards,
                'done': dones
            }
            
            np.savez(os.path.join(temp_dir, f"train_data_{i}.npz"), **data)
        
        try:
            # Test dataset creation
            dataset = MDPDataset(
                directory=temp_dir,
                gamma=0.99,
                max_length=100,
                env_name="CartPole-v1"
            )
            
            print(f"  ✅ Dataset created: {len(dataset)} episodes")
            
            # Test data loading
            s, a, rtg, d, timesteps, mask = dataset[0]
            print(f"  ✅ Data shapes: s={s.shape}, a={a.shape}, rtg={rtg.shape}")
            
            # Test dataloader
            dataloader = create_mdp_dataloader(
                data_dir=temp_dir,
                env_name="CartPole-v1",
                batch_size=2,
                max_length=100
            )
            
            batch = next(iter(dataloader))
            print(f"  ✅ Dataloader batch: {len(batch)} tensors")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Dataset creation failed: {e}")
            return False


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        ("gymnasium", "gym"),
        ("stable_baselines3", "stable_baselines3"),
        ("torch", "torch"),
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("tqdm", "tqdm")
    ]
    
    for package_name, import_name in required_packages:
        try:
            exec(f"import {import_name}")
            print(f"  ✅ {package_name}: OK")
        except ImportError as e:
            print(f"  ❌ {package_name}: {e}")
            return False
    
    return True


def test_device_availability():
    """Test device availability."""
    print("\nTesting device availability...")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing MDP setup...")
    print("=" * 50)
    
    tests = [
        ("Package imports", test_imports),
        ("Device availability", test_device_availability), 
        ("Environment creation", test_environments),
        ("Dataset creation", test_dataset_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        icon = "✅" if success else "❌"
        print(f"{icon} {test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed! MDP setup is ready.")
        print(f"\nNext steps:")
        print(f"  1. Run training: python train_online_rl.py --env CartPole-v1")
        print(f"  2. Collect data: python collect_mdp_datasets.py --env CartPole-v1")
        print(f"  3. Run full pipeline: python run_full_pipeline.py --env CartPole-v1")
    else:
        print(f"\n❌ Some tests failed. Please fix the issues before proceeding.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
