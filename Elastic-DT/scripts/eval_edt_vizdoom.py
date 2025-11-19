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


# Seeds for red and green pillars (same as in RATE)
reds = [
    2, 3, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20, 21, 25, 26, 27, 28, 29, 31, 38, 40, 41, 42, 45,
    46, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 61, 63, 64, 67, 68, 70, 72, 73, 74, 77, 80, 82, 84, 
    86, 88, 89, 90, 91, 92, 97, 98, 99, 100, 101, 103, 106, 108, 109, 113, 115, 116, 117, 120, 123, 
    124, 125, 126, 127, 128, 129, 133, 134, 136, 139, 140, 142, 144, 145, 147, 148, 151, 152, 153, 
    154, 156, 157, 158, 159, 161, 164, 165, 170, 171, 173
]

greens = [
    0, 1, 4, 5, 7, 12, 13, 19, 22, 23, 24, 30, 32, 33, 34, 35, 36, 37, 39, 43, 44, 47, 48, 56, 57,
    62, 65, 66, 69, 71, 75, 76, 78, 79, 81, 83, 85, 87, 93, 94, 95, 96, 102, 104, 105, 107, 110, 111, 
    112, 114, 118, 119, 121, 122, 130, 131, 132, 135, 137, 138, 141, 143, 146, 149, 150, 155, 160, 162, 
    163, 166, 167, 168, 169, 172, 175, 176, 177, 182, 183, 187, 190, 192, 193, 195, 199, 204, 206, 208, 
    209, 210, 212, 215, 216, 218, 219, 220, 221, 223, 224, 225
]


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
    print(f"Context length: {args.context_len}")
    print(f"Episode timeout: {args.episode_timeout}")
    print(f"Use separate red/green seeds: {args.use_pillar_seeds}")
    print("=" * 60)
    
    FRAME_SKIP = 2
    
    def evaluate_pillar(color, seeds):
        """Evaluate on a specific pillar color."""
        returns = []
        lengths = []
        
        for i, seed in enumerate(seeds):
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
            
            returns.append(episode_return)
            episode_length *= FRAME_SKIP
            lengths.append(episode_length)
            
            print(f"  {color.upper()} [{i+1}/{len(seeds)}] seed={seed}: "
                  f"Return={episode_return:.2f}, Length={episode_length}")
        
        return returns, lengths
    
    # Choose evaluation mode
    if args.use_pillar_seeds:
        # Use separate red and green seeds (like in RATE)
        SKIP_RETURN = 4
        
        # RED PILLAR
        print("\n" + "=" * 60)
        print("Evaluating RED pillar")
        print("=" * 60)
        seeds_red = reds[::SKIP_RETURN]
        red_returns, red_lengths = evaluate_pillar("red", seeds_red)
        
        # GREEN PILLAR
        print("\n" + "=" * 60)
        print("Evaluating GREEN pillar")
        print("=" * 60)
        seeds_green = greens[::SKIP_RETURN]
        green_returns, green_lengths = evaluate_pillar("green", seeds_green)
        
        # Combine results
        all_returns = red_returns + green_returns
        all_lengths = red_lengths + green_lengths
        
        # Per-color statistics
        red_mean = np.mean(red_returns)
        red_std = np.std(red_returns)
        green_mean = np.mean(green_returns)
        green_std = np.std(green_returns)
        
    else:
        # Simple sequential evaluation
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
            
            print(f"Episode {ep + 1}/{args.num_eval_episodes} seed={seed}: "
                  f"Return={episode_return:.2f}, Length={episode_length}")
    
    # Overall statistics
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    mean_length = np.mean(all_lengths)
    
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    
    if args.use_pillar_seeds:
        print(f"\nRED pillar ({len(red_returns)} episodes):")
        print(f"  Mean return: {red_mean:.2f} +/- {red_std:.2f}")
        print(f"  Min/Max: {np.min(red_returns):.2f} / {np.max(red_returns):.2f}")
        
        print(f"\nGREEN pillar ({len(green_returns)} episodes):")
        print(f"  Mean return: {green_mean:.2f} +/- {green_std:.2f}")
        print(f"  Min/Max: {np.min(green_returns):.2f} / {np.max(green_returns):.2f}")
        
        print(f"\nOVERALL ({len(all_returns)} episodes):")
    else:
        print(f"\nOVERALL ({len(all_returns)} episodes):")
    
    print(f"  Mean return: {mean_return:.2f} +/- {std_return:.2f}")
    print(f"  Mean episode length: {mean_length:.2f}")
    print(f"  Min/Max return: {np.min(all_returns):.2f} / {np.max(all_returns):.2f}")
    print("=" * 60)
    
    # Save results
    if args.save_results:
        results_path = checkpoint_path.replace(".pt", "_eval_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Target return: {args.target_return}\n")
            f.write(f"Context length: {args.context_len}\n")
            f.write(f"Use pillar seeds: {args.use_pillar_seeds}\n")
            f.write(f"Number of episodes: {len(all_returns)}\n")
            f.write(f"\n")
            
            if args.use_pillar_seeds:
                f.write(f"RED pillar ({len(red_returns)} episodes):\n")
                f.write(f"  Mean return: {red_mean:.2f} +/- {red_std:.2f}\n")
                f.write(f"  Min/Max: {np.min(red_returns):.2f} / {np.max(red_returns):.2f}\n")
                f.write(f"\n")
                f.write(f"GREEN pillar ({len(green_returns)} episodes):\n")
                f.write(f"  Mean return: {green_mean:.2f} +/- {green_std:.2f}\n")
                f.write(f"  Min/Max: {np.min(green_returns):.2f} / {np.max(green_returns):.2f}\n")
                f.write(f"\n")
            
            f.write(f"OVERALL:\n")
            f.write(f"  Mean return: {mean_return:.2f} +/- {std_return:.2f}\n")
            f.write(f"  Mean episode length: {mean_length:.2f}\n")
            f.write(f"  Min/Max return: {np.min(all_returns):.2f} / {np.max(all_returns):.2f}\n")
            f.write(f"\n")
            
            if args.use_pillar_seeds:
                f.write("Red pillar returns:\n")
                for i, ret in enumerate(red_returns):
                    f.write(f"  Episode {i + 1} (seed {seeds_red[i]}): {ret:.2f}\n")
                f.write("\nGreen pillar returns:\n")
                for i, ret in enumerate(green_returns):
                    f.write(f"  Episode {i + 1} (seed {seeds_green[i]}): {ret:.2f}\n")
            else:
                f.write("All returns:\n")
                for i, ret in enumerate(all_returns):
                    f.write(f"  Episode {i + 1}: {ret:.2f}\n")
        
        print(f"\nResults saved to: {results_path}")


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
    parser.add_argument("--num_eval_episodes", type=int, default=10,
                        help="Number of episodes (only used if --use_pillar_seeds is False)")
    parser.add_argument("--episode_timeout", type=int, default=150)
    parser.add_argument("--use_argmax", action="store_true",
                        help="Use argmax for action selection instead of sampling")
    parser.add_argument("--use_pillar_seeds", action="store_true",
                        help="Use separate red/green pillar seeds (like RATE)")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_results", action="store_true")
    
    args = parser.parse_args()
    
    evaluate(args)

