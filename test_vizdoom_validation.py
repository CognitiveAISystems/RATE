#!/usr/bin/env python3
"""
Test script for MATL validation on ViZDoom with loaded checkpoint
"""

import torch
import json
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from src.validation.val_vizdoom_two_colors import get_returns_VizDoom
from offline_rl_baselines.MATL import MATLModel

def load_checkpoint_and_config(checkpoint_dir):
    """Load checkpoint and config from directory"""
    checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model_config = config["model"]
    model = MATLModel(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    return model, config

def test_validation():
    """Test validation with loaded checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and config
    model, config = load_checkpoint_and_config("test_ckpt")
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Model mode: {config.get('model_mode', 'Unknown')}")
    print(f"Sequence format: {config['model'].get('sequence_format', 'Unknown')}")
    print(f"Memory size: {config['model'].get('memory_size', 'Unknown')}")
    
    # Test validation parameters
    ret = config["online_inference"]["desired_return_1"]
    seed = 42
    episode_timeout = config["online_inference"]["episode_timeout"]
    context_length = config["training"]["context_length"]
    use_argmax = config["online_inference"]["use_argmax"]
    
    print(f"\nValidation parameters:")
    print(f"Target return: {ret}")
    print(f"Episode timeout: {episode_timeout}")
    print(f"Context length: {context_length}")
    print(f"Use argmax: {use_argmax}")
    
    # Debug: Check model's RTG encoder
    print(f"\nModel RTG encoder input size: {model.ret_emb[0].in_features}")
    print(f"Model RTG encoder output size: {model.ret_emb[0].out_features}")
    
    # Run validation
    print(f"\nRunning validation...")
    try:
        episode_return, episode_length = get_returns_VizDoom(
            model=model,
            ret=ret,
            seed=seed,
            episode_timeout=episode_timeout,
            context_length=context_length,
            device=device,
            config=config,
            use_argmax=use_argmax,
            create_video=False
        )
        
        print(f"Validation completed!")
        print(f"Episode return: {episode_return}")
        print(f"Episode length: {episode_length}")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_validation()
