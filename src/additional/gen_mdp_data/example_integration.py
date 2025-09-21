"""
Example integration of MDP datasets with RATE/MATL training pipeline.

This script demonstrates how to use the collected MDP datasets with the existing
RATE/MATL framework for training transformer models on MDP tasks.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
import json

from src.envs_datasets.mdp_dataset import MDPDataset
from src.trainer import Trainer


def create_mdp_config(env_name: str, data_dir: str) -> dict:
    """Create a configuration for training MATL on MDP data."""
    
    # Load dataset metadata to get dimensions
    metadata_path = Path(data_dir) / "dataset_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        stats = metadata.get('statistics', {})
        state_dim = len(stats.get('obs_mean', [4]))  # Default to 4 for CartPole
        
        # Determine action dimension and type
        if 'action_distribution' in stats:
            # Discrete actions
            action_dim = stats.get('n_unique_actions', 2)
            action_type = 'discrete'
        else:
            # Continuous actions  
            action_mean = stats.get('action_mean', 0)
            if isinstance(action_mean, list):
                action_dim = len(action_mean)
            else:
                action_dim = 1
            action_type = 'continuous'
    else:
        # Fallback defaults
        state_dim = 4
        action_dim = 2 if env_name == "CartPole-v1" else 1
        action_type = 'discrete' if env_name in ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"] else 'continuous'
    
    config = {
        "model_mode": "MATL",
        "dtype": "float32",
        "model": {
            "env_name": env_name,
            "state_dim": state_dim,
            "act_dim": action_dim,
            "action_type": action_type,
            "max_length": 1000,
            "max_ep_len": 1000,
            "hidden_size": 128,
            "n_layer": 4,
            "n_head": 4,
            "n_inner": 256,
            "activation_function": "gelu",
            "n_positions": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "memory_size": 64,
            "memory_sharing_mode": "shared",
            "use_relative_bias": True,
            "use_tok2mem": True,
            "use_mem2tok": True,
            "use_lru": True,
            "lru_blend_alpha": 0.1,
            "detach_memory": False,
            "sequence_format": "sra"
        },
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "beta_1": 0.9,
            "beta_2": 0.95,
            "grad_norm_clip": 1.0,
            "context_length": 100,
            "sections": 1,
            "batch_size": 32,
            "epochs": 50,
            "warmup_steps": 1000,
            "log_last_segment_loss_only": True,
            "use_cosine_decay": True,
            "ckpt_epoch": 10,
            "online_inference": False
        },
        "wandb": {
            "wwandb": False,
            "project": f"MATL-MDP-{env_name}",
            "name": f"matl-{env_name.lower()}"
        }
    }
    
    return config


def create_mdp_dataloader(data_dir: str, env_name: str, config: dict) -> DataLoader:
    """Create a DataLoader for MDP data compatible with RATE/MATL."""
    
    dataset = MDPDataset(
        directory=data_dir,
        gamma=0.99,
        max_length=config["training"]["context_length"],
        env_name=env_name
    )
    
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )


def train_matl_on_mdp(env_name: str, data_dir: str):
    """Train MATL model on MDP dataset."""
    
    print(f"Training MATL on {env_name} dataset")
    print(f"Data directory: {data_dir}")
    
    # Create configuration
    config = create_mdp_config(env_name, data_dir)
    
    # Create data loader
    train_dataloader = create_mdp_dataloader(data_dir, env_name, config)
    
    print(f"Dataset size: {len(train_dataloader.dataset)} episodes")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Context length: {config['training']['context_length']}")
    
    # Create trainer
    with Trainer(config) as trainer:
        # Train the model
        model = trainer.train(train_dataloader)
        print("Training completed!")
        
        return model


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MATL on MDP datasets")
    parser.add_argument("--env", type=str, required=True,
                       choices=["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", 
                               "Acrobot-v1", "Pendulum-v1"],
                       help="Environment name")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing the MDP dataset")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Dataset directory not found: {args.data_dir}")
        print("Please run data collection first:")
        print(f"  python collect_mdp_datasets.py --env {args.env}")
        return
    
    # Train model
    try:
        model = train_matl_on_mdp(args.env, args.data_dir)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
