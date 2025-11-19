import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import argparse

# Add parent directory and RATE src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from decision_transformer.vizdoom_model import ElasticDecisionTransformerViZDoom
from decision_transformer.vizdoom_inference import ViZDoomEDTWrapper
from src.validation.val_vizdoom_two_colors import get_returns_VizDoom


def evaluate(args):
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    edt_model = ElasticDecisionTransformerViZDoom(
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
    
    # Load checkpoint
    checkpoint_path = args.checkpoint_path
    print(f"Loading checkpoint from: {checkpoint_path}")
    edt_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    edt_model.eval()
    
    # Wrap model for compatibility with validation interface
    model = ViZDoomEDTWrapper(edt_model)
    
    print(f"Model loaded successfully")
    
    # Create evaluation config
    eval_config = {
        "dtype": "float32",
        "model_mode": "DT",  # Use DT mode for inference
    }
    
    # Run evaluation
    print("=" * 60)
    print(f"Evaluating on ViZDoom Two Colors")
    print(f"Target return: {args.target_return}")
    print(f"Number of episodes: {args.num_eval_episodes}")
    print(f"Episode timeout: {args.episode_timeout}")
    print(f"Context length: {args.context_len}")
    print("=" * 60)
    
    all_returns = []
    all_lengths = []
    
    for ep in range(args.num_eval_episodes):
        seed = args.seed + ep
        
        episode_return, episode_length = get_returns_VizDoom(
            model=model,
            ret=args.target_return,
            seed=seed,
            episode_timeout=args.episode_timeout,
            context_length=args.context_len,
            device=device,
            config=eval_config,
            use_argmax=args.use_argmax,
            create_video=False
        )
        
        all_returns.append(episode_return)
        all_lengths.append(episode_length)
        
        print(f"Episode {ep + 1}/{args.num_eval_episodes}: "
              f"Return = {episode_return:.2f}, Length = {episode_length}")
    
    # Print statistics
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    mean_length = np.mean(all_lengths)
    
    print("=" * 60)
    print("Evaluation Results:")
    print(f"Mean return: {mean_return:.2f} +/- {std_return:.2f}")
    print(f"Mean episode length: {mean_length:.2f}")
    print(f"Min return: {np.min(all_returns):.2f}")
    print(f"Max return: {np.max(all_returns):.2f}")
    print("=" * 60)
    
    # Save results
    if args.save_results:
        results_path = checkpoint_path.replace(".pt", "_eval_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Target return: {args.target_return}\n")
            f.write(f"Context length: {args.context_len}\n")
            f.write(f"Number of episodes: {args.num_eval_episodes}\n")
            f.write(f"\n")
            f.write(f"Mean return: {mean_return:.2f} +/- {std_return:.2f}\n")
            f.write(f"Mean episode length: {mean_length:.2f}\n")
            f.write(f"Min return: {np.min(all_returns):.2f}\n")
            f.write(f"Max return: {np.max(all_returns):.2f}\n")
            f.write(f"\n")
            f.write("All returns:\n")
            for i, ret in enumerate(all_returns):
                f.write(f"  Episode {i + 1}: {ret:.2f}\n")
        
        print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--context_len", type=int, default=50)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--act_dim", type=int, default=5)
    
    # EDT specific
    parser.add_argument("--num_bin", type=int, default=60)
    parser.add_argument("--dt_mask", action="store_true")
    parser.add_argument("--real_rtg", action="store_true")
    parser.add_argument("--rtg_scale", type=int, default=1000)
    
    # Evaluation
    parser.add_argument("--target_return", type=float, default=56.5,
                        help="Target return for evaluation")
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--episode_timeout", type=int, default=150)
    parser.add_argument("--use_argmax", action="store_true",
                        help="Use argmax for action selection instead of sampling")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_results", action="store_true")
    
    args = parser.parse_args()
    
    evaluate(args)

