"""
ARShot Environment Validation

This module provides validation functionality for the ARShot environment.
It includes functions for running inference and evaluating model performance
on associative retrieval tasks with 'shot' queries.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm

from src.envs.associative_retrieval.arshotenv import ARShotEnv
from src.utils.set_seed import set_seed


@torch.no_grad()
def sample(
    model, x, block_size, steps, sample=False, top_k=None, actions=None, 
    rtgs=None, timestep=None, mem_tokens=1, saved_context=None, hidden=None,
    memory_states=None, pos_offset=0
):
    """Sample from the model for ARShot environment.
    
    Args:
        model: The model to sample from.
        x: Input observations.
        block_size: Maximum context length.
        steps: Number of steps to sample.
        sample: Whether to sample or use greedy decoding.
        top_k: Top-k sampling parameter.
        actions: Previous actions.
        rtgs: Return-to-go values.
        timestep: Current timestep.
        mem_tokens: Memory tokens.
        saved_context: Saved context from previous steps.
        hidden: Hidden state for LSTM models.
        memory_states: Memory states for ELMUR models.
        pos_offset: Position offset for ELMUR models.
    
    Returns:
        Tuple containing logits, memory tokens, memory, attention map, hidden state, and memory states.
    """
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:]
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:]
        
        if saved_context is not None:
            results = model(
                x_cond, actions, rtgs, None, timestep, *saved_context, mem_tokens=mem_tokens, 
                hidden=hidden, memory_states=memory_states, pos_offset=pos_offset)
        else:
            results = model(
                x_cond, actions, rtgs, None, timestep, mem_tokens=mem_tokens, 
                hidden=hidden, memory_states=memory_states, pos_offset=pos_offset)

        logits = results['logits'][:, -1, :]
        memory = results.get('new_mems', None)
        mem_tokens = results.get('mem_tokens', None)
        hidden = results.get('hidden', None)
        attn_map = getattr(model, 'attn_map', None)
        memory_states = results.get('memory_states', None)
        
    return logits, mem_tokens, memory, attn_map, hidden, memory_states


def get_returns_ARShot(
    model, ret: float, seeds: List[int], n_pairs: int, shot_mode: str,
    episode_timeout: int, context_length: int, device: torch.device, 
    config: Dict[str, Any], create_video: bool = False, use_argmax: bool = True,
    **env_kwargs
) -> Tuple[List[float], int]:
    """Evaluate model on ARShot environment.
    
    Args:
        model: The model to evaluate.
        ret: Target return value.
        seeds: List of random seeds for evaluation.
        n_pairs: Number of key-value pairs in the environment.
        shot_mode: Mode for shot placement ("after_pairs" or "after_any_colon").
        episode_timeout: Maximum episode length.
        context_length: Context length for the model.
        device: Device to run evaluation on.
        config: Configuration dictionary.
        create_video: Whether to create videos (not implemented).
        use_argmax: Whether to use argmax for action selection.
        **env_kwargs: Additional environment arguments.
    
    Returns:
        Tuple of (episode_returns, num_successes).
    """
    set_seed(42)
    batch_size = len(seeds)
    
    # Get dtype from config
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(config["dtype"], torch.float32)

    # Create environments
    envs = []
    for i in tqdm(range(batch_size), desc=f"Creating environments (n_pairs={n_pairs}, mode={shot_mode})"):
        env = ARShotEnv(
            n_pairs=n_pairs,
            shot_mode=shot_mode,
            rng_seed=seeds[i],
            **env_kwargs
        )
        envs.append(env)
    
    # Initialize states
    states = []
    infos = []
    for env in envs:
        obs, info = env.reset()
        states.append(obs)
        infos.append(info)
    
    # Convert to tensor
    states = torch.tensor(states, dtype=torch.long, device=device).unsqueeze(1)  # (batch, 1)
    
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    episode_rewards = torch.zeros(batch_size, device=device, dtype=dtype)
    successes = 0
    
    actions = torch.zeros((batch_size, 0), device=device, dtype=torch.long)
    target_return = torch.full((batch_size, 1), ret, device=device, dtype=dtype)
    timesteps = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
    
    # Model-specific initialization
    is_lstm = hasattr(model, 'backbone') and model.backbone in ['lstm', 'gru']
    
    mem_tokens = model.mem_tokens.repeat(1, batch_size, 1).detach() if hasattr(model, 'mem_tokens') and model.mem_tokens is not None else None
    saved_context = None
    hidden = model.reset_hidden(batch_size, device) if is_lstm else None
    memory_states = model.init_memory(batch_size, device) if config["model_mode"] == "ELMUR" else None
    
    # Initialize variables for context management
    new_mem_tokens = mem_tokens
    new_context = saved_context
    new_memory_states = memory_states
    
    for t in tqdm(range(episode_timeout), desc=f"ARShot Inference (n_pairs={n_pairs}, mode={shot_mode})"):
        # Add new action slot
        actions = torch.cat([actions, torch.zeros((batch_size, 1), device=device, dtype=torch.long)], dim=1)
        
        # Context length management
        if not is_lstm and actions.shape[1] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[:, slice_index:]
            states = states[:, slice_index:]
            target_return = target_return[:, slice_index:]
            timesteps = timesteps[:, slice_index:]
            if t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context
                if config["model_mode"] == "ELMUR":
                    memory_states = new_memory_states
        
        # Prepare inputs based on model type
        if is_lstm:
            states_to_pass = states[:, -1:].unsqueeze(-1)  # (batch, 1, 1)
            act_to_pass = None if t == 0 else actions[:, -1:].unsqueeze(-1)
            rtg_to_pass = target_return[:, -1:].unsqueeze(-1)
            timesteps_to_pass = timesteps[:, -1:]
        else:
            states_to_pass = states.unsqueeze(-1)  # Add feature dimension
            act_to_pass = None if t == 0 else actions[:, 1:].unsqueeze(-1)
            rtg_to_pass = target_return.unsqueeze(-1)
            timesteps_to_pass = timesteps
            if act_to_pass is not None and act_to_pass.shape[1] == 0:
                act_to_pass = None
        
        # Calculate position offset for ELMUR
        if config["model_mode"] == "ELMUR":
            segment_idx = t // context_length
            sequence_format = getattr(model, 'sequence_format', 'sra')
            multiplier = model.get_sequence_length_multiplier()
            pos_offset_val = segment_idx * context_length * multiplier
        else:
            window_len = min(context_length, t + 1)
            pos_offset_val = (t - window_len + 1) * 3
        
        # Sample from model
        sample_outputs = sample(
            model=model,
            x=states_to_pass,
            block_size=context_length,
            steps=1,
            sample=not use_argmax,
            actions=act_to_pass,
            rtgs=rtg_to_pass,
            timestep=timesteps_to_pass,
            mem_tokens=mem_tokens,
            saved_context=saved_context,
            hidden=hidden,
            memory_states=memory_states,
            pos_offset=pos_offset_val
        )
        
        sampled_action, new_mem_tokens, new_context, attn_map, new_hidden, new_memory_states = sample_outputs
        
        if is_lstm:
            hidden = new_hidden
        
        # Get actions
        if use_argmax:
            predicted_actions = torch.argmax(sampled_action, dim=-1)
        else:
            action_probs = torch.softmax(sampled_action, dim=-1)
            predicted_actions = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        
        actions[:, -1] = predicted_actions
        
        # Step environments
        next_states = []
        for i, env in enumerate(envs):
            if done[i]:
                next_states.append(states[i, -1].item())
                continue
            
            action_id = predicted_actions[i].item()
            next_obs, reward, terminated, truncated, next_info = env.step(action_id)
            next_states.append(next_obs)
            
            episode_rewards[i] += reward
            done[i] = terminated or truncated
        
        # Update states and timesteps
        next_states = torch.tensor(next_states, dtype=torch.long, device=device).unsqueeze(1)
        states = torch.cat([states, next_states], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((batch_size, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
        
        # Update target return (no reward decay in ARShot)
        pred_return = target_return[:, -1:] - 0  # No reward during episode except at the end
        target_return = torch.cat([target_return, pred_return], dim=1)
        
        # Check if all episodes are done
        if done.all():
            break
    
    # Count successes (reward == 1.0 means correct answer)
    successes = (episode_rewards == 1.0).sum().item()
    
    # Convert to list for compatibility
    episode_returns = episode_rewards.cpu().float().tolist()
    
    return episode_returns, successes


def evaluate_arshot_model(
    model, config: Dict[str, Any], device: torch.device,
    n_pairs_list: List[int] = [6, 10, 15, 20],
    shot_modes: List[str] = ["after_pairs", "after_any_colon"],
    num_seeds: int = 10, episode_timeout: int = 200
) -> Dict[str, Any]:
    """Comprehensive evaluation of model on ARShot tasks.
    
    Args:
        model: The model to evaluate.
        config: Configuration dictionary.
        device: Device to run evaluation on.
        n_pairs_list: List of n_pairs values to test.
        shot_modes: List of shot modes to test.
        num_seeds: Number of random seeds per configuration.
        episode_timeout: Maximum episode length.
    
    Returns:
        Dictionary containing evaluation results.
    """
    results = {}
    
    for n_pairs in n_pairs_list:
        for shot_mode in shot_modes:
            key = f"n_pairs_{n_pairs}_{shot_mode}"
            seeds = list(range(num_seeds))
            
            episode_returns, successes = get_returns_ARShot(
                model=model,
                ret=1.0,
                seeds=seeds,
                n_pairs=n_pairs,
                shot_mode=shot_mode,
                episode_timeout=episode_timeout,
                context_length=config["training"]["context_length"],
                device=device,
                config=config,
                deterministic_vocab=True,
                full_universe_vocab=True,
                randomize_pairs=True,
                include_pass_token=True
            )
            
            success_rate = successes / len(seeds)
            mean_return = np.mean(episode_returns)
            
            results[key] = {
                'success_rate': success_rate,
                'mean_return': mean_return,
                'successes': successes,
                'total_episodes': len(seeds),
                'episode_returns': episode_returns
            }
            
            print(f"{key}: Success rate = {success_rate:.3f}, Mean return = {mean_return:.3f}")
    
    return results


if __name__ == "__main__":
    # Test the validation code
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
    
    # Create test environment to get vocab size
    env = ARShotEnv(n_pairs=6, shot_mode="after_pairs")
    dummy_model = DummyModel(len(env.vocab))
    
    # Test configuration
    config = {
        "model_mode": "DT",
        "dtype": "float32",
        "training": {"context_length": 50}
    }
    
    device = torch.device('cpu')
    
    # Run evaluation
    print("Testing ARShot validation...")
    episode_returns, successes = get_returns_ARShot(
        model=dummy_model,
        ret=1.0,
        seeds=[1, 2, 3],
        n_pairs=6,
        shot_mode="after_pairs",
        episode_timeout=50,
        context_length=50,
        device=device,
        config=config
    )
    
    print(f"Episode returns: {episode_returns}")
    print(f"Successes: {successes}")
    print("ARShot validation test completed!")
