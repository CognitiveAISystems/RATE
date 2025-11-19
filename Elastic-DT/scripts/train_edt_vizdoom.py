import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from decision_transformer.vizdoom_model import ElasticDecisionTransformerViZDoom
from decision_transformer.vizdoom_dataset import ViZDoomTrajectoryDataset
from decision_transformer.utils import encode_return
from torch.utils.data import DataLoader
import argparse


def train(args):
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Dataset
    dataset_path = args.dataset_dir
    print(f"Loading dataset from: {dataset_path}")
    
    train_dataset = ViZDoomTrajectoryDataset(
        dataset_dir=dataset_path,
        context_len=args.context_len,
        rtg_scale=args.rtg_scale,
        normalize=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )
    
    # Model
    model = ElasticDecisionTransformerViZDoom(
        img_channels=3,
        img_height=64,
        img_width=112,
        act_dim=args.act_dim,
        n_blocks=args.n_blocks,
        h_dim=args.embed_dim,
        context_len=args.context_len,
        n_heads=args.n_heads,
        drop_p=args.dropout_p,
        env_name="vizdoom",
        num_bin=args.num_bin,
        dt_mask=args.dt_mask,
        rtg_scale=args.rtg_scale,
        real_rtg=args.real_rtg,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wt_decay
    )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    
    # Training loop
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    
    # Create checkpoint directory
    os.makedirs(args.chk_pt_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Starting training at: {start_time_str}")
    print(f"Training for {args.max_train_iters} iterations")
    print(f"Updates per iteration: {args.num_updates_per_iter}")
    print("=" * 60)
    
    total_updates = 0
    data_iter = iter(train_dataloader)
    
    for i_train_iter in tqdm(range(1, args.max_train_iters + 1)):
        model.train()
        
        log_action_losses = []
        log_exp_losses = []
        ret_ce_losses = []
        
        for _ in tqdm(range(args.num_updates_per_iter), leave=False):
            try:
                (timesteps, states, actions, returns_to_go, rewards, traj_mask) = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                (timesteps, states, actions, returns_to_go, rewards, traj_mask) = next(data_iter)
            
            # Move to device
            timesteps = timesteps.to(device)
            states = states.to(device)
            actions = actions.to(device)
            returns_to_go = returns_to_go.to(device)
            rewards = rewards.to(device)
            traj_mask = traj_mask.to(device)
            
            # Forward pass
            (
                _,
                action_preds,
                return_preds,
                imp_return_preds,
                reward_preds,
            ) = model(
                timesteps=timesteps,
                states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                rewards=rewards,
            )
            
            # Action loss (cross-entropy for discrete actions)
            action_preds_flat = action_preds.view(-1, args.act_dim)[traj_mask.view(-1) > 0]
            action_target_flat = actions.view(-1)[traj_mask.view(-1) > 0]
            action_loss = F.cross_entropy(action_preds_flat, action_target_flat.long())
            
            # Return expectile loss
            def expectile_loss(diff, expectile=0.8):
                weight = torch.where(diff > 0, expectile, (1 - expectile))
                return weight * (diff ** 2)
            
            imp_return_pred = imp_return_preds.reshape(-1, 1)[traj_mask.view(-1) > 0]
            imp_return_target = returns_to_go.reshape(-1, 1)[traj_mask.view(-1) > 0]
            imp_loss = expectile_loss(
                (imp_return_target - imp_return_pred), args.expectile
            ).mean()
            
            # Return cross-entropy loss
            return_preds_flat = return_preds.reshape(-1, int(args.num_bin))[traj_mask.view(-1) > 0]
            return_target_flat = (
                encode_return(
                    "vizdoom",
                    returns_to_go,
                    num_bin=args.num_bin,
                    rtg_scale=args.rtg_scale,
                )
                .float()
                .reshape(-1, 1)[traj_mask.view(-1) > 0]
            )
            
            def cross_entropy(logits, labels):
                labels = F.one_hot(labels.long(), num_classes=int(args.num_bin)).squeeze()
                criterion = nn.CrossEntropyLoss()
                return criterion(logits, labels.float())
            
            ret_ce_loss = cross_entropy(return_preds_flat, return_target_flat)
            
            # Total loss
            loss = (
                action_loss
                + imp_loss * args.exp_loss_weight
                + args.ce_weight * ret_ce_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
            
            # Log
            log_action_losses.append(action_loss.detach().cpu().item())
            log_exp_losses.append(imp_loss.detach().cpu().item())
            ret_ce_losses.append(ret_ce_loss.detach().cpu().item())
        
        total_updates += args.num_updates_per_iter
        
        # Log statistics
        mean_action_loss = np.mean(log_action_losses)
        mean_expectile_loss = np.mean(log_exp_losses)
        mean_ret_loss = np.mean(ret_ce_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
        
        log_str = (
            "=" * 60 + "\n"
            + f"Iteration: {i_train_iter}/{args.max_train_iters}\n"
            + f"Time elapsed: {time_elapsed}\n"
            + f"Total updates: {total_updates}\n"
            + f"Action loss: {mean_action_loss:.5f}\n"
            + f"Expectile loss: {mean_expectile_loss:.5f}\n"
            + f"Return CE loss: {mean_ret_loss:.5f}\n"
        )
        
        if i_train_iter % 10 == 0:
            print(log_str)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "train/action_loss": mean_action_loss,
                "train/expectile_loss": mean_expectile_loss,
                "train/return_ce_loss": mean_ret_loss,
                "train/total_updates": total_updates,
            }, step=i_train_iter)
        
        # Save checkpoint
        if i_train_iter % args.model_save_iters == 0:
            save_path = os.path.join(
                args.chk_pt_dir,
                f"edt_vizdoom_iter_{i_train_iter}_seed_{args.seed}.pt"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")
    
    # Save final model
    final_save_path = os.path.join(
        args.chk_pt_dir,
        f"edt_vizdoom_final_seed_{args.seed}.pt"
    )
    torch.save(model.state_dict(), final_save_path)
    
    print("=" * 60)
    print("Training completed!")
    print(f"Final model saved to: {final_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--dataset_dir", type=str, default="data/ViZDoom_Two_Colors_150/")
    parser.add_argument("--rtg_scale", type=int, default=1000)
    
    # Model
    parser.add_argument("--context_len", type=int, default=50)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--act_dim", type=int, default=5)
    
    # EDT specific
    parser.add_argument("--ce_weight", type=float, default=0.001)
    parser.add_argument("--num_bin", type=int, default=60)
    parser.add_argument("--dt_mask", action="store_true")
    parser.add_argument("--expectile", type=float, default=0.99)
    parser.add_argument("--exp_loss_weight", type=float, default=0.5)
    parser.add_argument("--real_rtg", action="store_true")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wt_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--max_train_iters", type=int, default=500)
    parser.add_argument("--num_updates_per_iter", type=int, default=100)
    parser.add_argument("--model_save_iters", type=int, default=50)
    
    # Other
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chk_pt_dir", type=str, default="checkpoints/vizdoom/")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project_name", type=str, default="EDT-ViZDoom")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.project_name, config=vars(args))
    
    train(args)

