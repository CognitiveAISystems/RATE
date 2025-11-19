"""
Complete workflow example for training and evaluating Elastic-DT on ViZDoom.

This script demonstrates how to:
1. Load the dataset
2. Create and train the model
3. Save checkpoints
4. Load and evaluate the model

Run this script from the RATE root directory:
    python Elastic-DT/example_vizdoom_workflow.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from decision_transformer.vizdoom_model import ElasticDecisionTransformerViZDoom
from decision_transformer.vizdoom_dataset import ViZDoomTrajectoryDataset
from decision_transformer.vizdoom_inference import ViZDoomEDTWrapper
from src.validation.val_vizdoom_two_colors import get_returns_VizDoom


def main():
    # Configuration
    config = {
        "dataset_dir": "data/ViZDoom_Two_Colors_150/",
        "checkpoint_dir": "checkpoints/vizdoom_example/",
        "context_len": 50,
        "n_blocks": 4,
        "embed_dim": 128,
        "n_heads": 4,
        "dropout_p": 0.1,
        "act_dim": 5,
        "batch_size": 128,
        "lr": 1e-4,
        "num_train_steps": 100,  # Small number for example
        "save_every": 50,
        "rtg_scale": 1000,
        "num_bin": 60,
        "expectile": 0.99,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print("=" * 60)
    print("Elastic-DT on ViZDoom - Complete Workflow Example")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    device = torch.device(config["device"])
    print(f"\nUsing device: {device}")
    
    # ========================================
    # STEP 1: Load Dataset
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    
    if not os.path.exists(config["dataset_dir"]):
        print(f"ERROR: Dataset not found at {config['dataset_dir']}")
        print("Please generate the dataset first using:")
        print("    python src/additional/gen_vizdoom_data/gen_vizdoom_data.py")
        return
    
    train_dataset = ViZDoomTrajectoryDataset(
        dataset_dir=config["dataset_dir"],
        context_len=config["context_len"],
        rtg_scale=config["rtg_scale"],
        normalize=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )
    
    print(f"Dataset loaded: {len(train_dataset)} trajectories")
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of batches: {len(train_dataloader)}")
    
    # ========================================
    # STEP 2: Create Model
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: Creating Model")
    print("=" * 60)
    
    model = ElasticDecisionTransformerViZDoom(
        img_channels=3,
        img_height=64,
        img_width=112,
        act_dim=config["act_dim"],
        n_blocks=config["n_blocks"],
        h_dim=config["embed_dim"],
        context_len=config["context_len"],
        n_heads=config["n_heads"],
        drop_p=config["dropout_p"],
        env_name="vizdoom",
        num_bin=config["num_bin"],
        dt_mask=False,
        rtg_scale=config["rtg_scale"],
        real_rtg=False,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"Architecture:")
    print(f"  - Transformer blocks: {config['n_blocks']}")
    print(f"  - Hidden dimension: {config['embed_dim']}")
    print(f"  - Attention heads: {config['n_heads']}")
    print(f"  - Context length: {config['context_len']}")
    
    # ========================================
    # STEP 3: Train Model (Brief Example)
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 3: Training Model")
    print("=" * 60)
    print(f"Training for {config['num_train_steps']} steps (example only)")
    print("Note: For real training, use the training script for many more steps")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-4
    )
    
    model.train()
    data_iter = iter(train_dataloader)
    
    for step in range(config["num_train_steps"]):
        try:
            timesteps, states, actions, returns_to_go, rewards, traj_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            timesteps, states, actions, returns_to_go, rewards, traj_mask = next(data_iter)
        
        # Move to device
        timesteps = timesteps.to(device)
        states = states.to(device)
        actions = actions.to(device)
        returns_to_go = returns_to_go.to(device)
        traj_mask = traj_mask.to(device)
        
        # Forward pass
        _, action_preds, _, _, _ = model(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            rewards=None
        )
        
        # Simple loss (just action prediction for this example)
        action_preds_flat = action_preds.view(-1, config["act_dim"])[traj_mask.view(-1) > 0]
        action_target_flat = actions.view(-1)[traj_mask.view(-1) > 0]
        loss = torch.nn.functional.cross_entropy(action_preds_flat, action_target_flat.long())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{config['num_train_steps']}: Loss = {loss.item():.4f}")
    
    print("\nTraining complete!")
    
    # ========================================
    # STEP 4: Save Model
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 4: Saving Model")
    print("=" * 60)
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(
        config["checkpoint_dir"],
        f"edt_vizdoom_example_seed_{config['seed']}.pt"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")
    
    # ========================================
    # STEP 5: Evaluate Model
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating Model")
    print("=" * 60)
    
    # Load model (demonstrating how to load from checkpoint)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Model loaded from checkpoint")
    
    # Wrap model for compatibility with validation
    wrapped_model = ViZDoomEDTWrapper(model)
    
    eval_config = {
        "dtype": "float32",
        "model_mode": "DT",
    }
    
    print("\nRunning evaluation episodes...")
    num_eval_episodes = 3  # Small number for example
    target_return = 56.5
    
    all_returns = []
    for ep in range(num_eval_episodes):
        seed = config["seed"] + ep
        episode_return, episode_length = get_returns_VizDoom(
            model=wrapped_model,
            ret=target_return,
            seed=seed,
            episode_timeout=150,
            context_length=config["context_len"],
            device=device,
            config=eval_config,
            use_argmax=False,
            create_video=False
        )
        all_returns.append(episode_return)
        print(f"  Episode {ep + 1}: Return = {episode_return:.2f}, Length = {episode_length}")
    
    mean_return = np.mean(all_returns)
    print(f"\nMean return over {num_eval_episodes} episodes: {mean_return:.2f}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. For full training, use: bash Elastic-DT/scripts/run_vizdoom_train.sh")
    print("2. For full evaluation, use: bash Elastic-DT/scripts/run_vizdoom_eval.sh")
    print("3. See VIZDOOM_QUICKSTART.md for more details")
    print("\nCheckpoint saved at:", checkpoint_path)
    print("=" * 60)


if __name__ == "__main__":
    main()

