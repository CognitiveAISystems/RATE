#!/usr/bin/env python3
"""
Test script for ARShot integration with RATE training pipeline.
This script tests the dataset, encoders, and basic functionality.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_arshot_environment():
    """Test the ARShot environment."""
    print("Testing ARShot Environment...")
    
    from src.envs.associative_retrieval.arshotenv import ARShotEnv
    
    env = ARShotEnv(
        n_pairs=6,
        shot_mode="after_pairs",
        deterministic_vocab=True,
        full_universe_vocab=True,
        randomize_pairs=True,
        include_pass_token=True
    )
    
    obs, info = env.reset()
    print(f"✓ Environment created successfully")
    print(f"  Vocab size: {env.observation_space.n}")
    print(f"  Query key: {info['query_key']}")
    print(f"  Stream length: {len(env.decode_stream())}")
    print(f"  First 10 tokens: {env.decode_stream()[:10]}")
    
    return env


def test_arshot_dataset():
    """Test the ARShot dataset."""
    print("\nTesting ARShot Dataset...")
    
    from src.envs_datasets.arshot_dataset import ARShotDataset
    
    dataset = ARShotDataset(
        n_pairs=6,
        shot_mode="after_pairs",
        max_length=50,
        num_episodes=10,
        deterministic_vocab=True,
        full_universe_vocab=True,
        randomize_pairs=True,
        include_pass_token=True
    )
    
    print(f"✓ Dataset created successfully")
    print(f"  Number of episodes: {len(dataset)}")
    print(f"  Vocab size: {dataset.vocab_size}")
    
    # Test getting an episode
    s, a, rtg, d, timesteps, mask = dataset[0]
    print(f"  Episode shapes:")
    print(f"    Observations: {s.shape}")
    print(f"    Actions: {a.shape}")
    print(f"    RTG: {rtg.shape}")
    print(f"    Done: {d.shape}")
    print(f"    Timesteps: {timesteps.shape}")
    print(f"    Mask: {mask.shape}")
    
    # Show first few tokens
    first_obs_tokens = [dataset.env.id_to_token[idx.item()] for idx in s[:10]]
    first_action_tokens = [dataset.env.id_to_token[idx.item()] for idx in a[:10]]
    print(f"  First 10 obs tokens: {first_obs_tokens}")
    print(f"  First 10 action tokens: {first_action_tokens}")
    
    return dataset


def test_arshot_encoders():
    """Test the ARShot encoders."""
    print("\nTesting ARShot Encoders...")
    
    from src.envs.associative_retrieval.arshotenv import ARShotEnv
    from RATE.env_encoders.arshot_encoders import create_arshot_encoders
    
    env = ARShotEnv(
        n_pairs=6,
        shot_mode="after_pairs",
        deterministic_vocab=True,
        full_universe_vocab=True,
        randomize_pairs=True,
        include_pass_token=True
    )
    
    d_model = 128
    encoders = create_arshot_encoders(env, d_model)
    
    print(f"✓ Encoders created successfully")
    print(f"  Available encoders: {list(encoders.keys())}")
    
    # Test token mapper
    token_mapper = encoders['token_mapper']
    vocab_info = token_mapper.get_vocab_info()
    print(f"  Vocab info: {vocab_info}")
    
    # Test encoding
    obs_encoder = encoders['obs_encoder']
    act_encoder = encoders['act_encoder']
    act_decoder = encoders['act_decoder']
    
    batch_size, seq_len = 2, 10
    dummy_obs = torch.randint(0, len(env.vocab), (batch_size, seq_len))
    dummy_actions = torch.randint(0, len(env.vocab), (batch_size, seq_len))
    
    obs_embeddings = obs_encoder(dummy_obs)
    act_embeddings = act_encoder(dummy_actions)
    logits = act_decoder(obs_embeddings)
    
    print(f"  Encoding test:")
    print(f"    Observation embeddings: {obs_embeddings.shape}")
    print(f"    Action embeddings: {act_embeddings.shape}")
    print(f"    Action logits: {logits.shape}")
    
    return encoders


def test_arshot_dataloader():
    """Test the ARShot dataloader integration."""
    print("\nTesting ARShot Dataloader Integration...")
    
    from src.utils.dataloaders import create_dataloader
    
    config = {
        "model": {
            "env_name": "arshot",
            "n_pairs": 6,
            "shot_mode": "after_pairs",
            "deterministic_vocab": True,
            "full_universe_vocab": True,
            "randomize_pairs": True,
            "include_pass_token": True
        },
        "data": {
            "gamma": 1.0,
            "num_episodes": 50
        },
        "training": {
            "batch_size": 4
        }
    }
    
    max_length = 50
    segment_length = 30
    
    try:
        dataloader = create_dataloader(config, max_length, segment_length)
        print(f"✓ Dataloader created successfully")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Dataset size: {len(dataloader.dataset)}")
        
        # Test getting a batch
        batch = next(iter(dataloader))
        s, a, rtg, d, timesteps, mask = batch
        print(f"  Batch shapes:")
        print(f"    Observations: {s.shape}")
        print(f"    Actions: {a.shape}")
        print(f"    RTG: {rtg.shape}")
        print(f"    Done: {d.shape}")
        print(f"    Timesteps: {timesteps.shape}")
        print(f"    Mask: {mask.shape}")
        
        return dataloader
        
    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        return None


def test_arshot_validation():
    """Test the ARShot validation code."""
    print("\nTesting ARShot Validation...")
    
    from src.validation.val_arshot import get_returns_ARShot
    from src.envs.associative_retrieval.arshotenv import ARShotEnv
    
    # Create a dummy model for testing
    class DummyModel:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.mem_tokens = None
            
        def eval(self):
            pass
            
        def __call__(self, x, actions, rtgs, targets, timesteps, **kwargs):
            batch_size, seq_len = x.shape[:2]
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
            return {'logits': logits}
    
    # Create test environment to get vocab size - use same config as dataset test
    env = ARShotEnv(
        n_pairs=6, 
        shot_mode="after_pairs",
        deterministic_vocab=True,
        full_universe_vocab=True,
        randomize_pairs=True,
        include_pass_token=True  # This is important!
    )
    dummy_model = DummyModel(len(env.vocab))
    
    # Test configuration
    config = {
        "model_mode": "DT",
        "dtype": "float32",
        "training": {"context_length": 50}
    }
    
    device = torch.device('cpu')
    
    try:
        episode_returns, successes = get_returns_ARShot(
            model=dummy_model,
            ret=1.0,
            seeds=[1, 2, 3],
            n_pairs=6,
            shot_mode="after_pairs",
            episode_timeout=50,
            context_length=50,
            device=device,
            config=config,
            deterministic_vocab=True,
            full_universe_vocab=True,
            randomize_pairs=True,
            include_pass_token=True  # Pass the same config as env
        )
        
        print(f"✓ Validation test completed successfully")
        print(f"  Episode returns: {episode_returns}")
        print(f"  Successes: {successes}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ARShot Integration Test Suite")
    print("=" * 60)
    
    success_count = 0
    total_tests = 5
    
    try:
        # Test 1: Environment
        env = test_arshot_environment()
        if env is not None:
            success_count += 1
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
    
    try:
        # Test 2: Dataset
        dataset = test_arshot_dataset()
        if dataset is not None:
            success_count += 1
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
    
    try:
        # Test 3: Encoders
        encoders = test_arshot_encoders()
        if encoders is not None:
            success_count += 1
    except Exception as e:
        print(f"✗ Encoders test failed: {e}")
    
    try:
        # Test 4: Dataloader
        dataloader = test_arshot_dataloader()
        if dataloader is not None:
            success_count += 1
    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
    
    try:
        # Test 5: Validation
        validation_success = test_arshot_validation()
        if validation_success:
            success_count += 1
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    print("=" * 60)
    
    if success_count == total_tests:
        print("🎉 All tests passed! ARShot integration is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
