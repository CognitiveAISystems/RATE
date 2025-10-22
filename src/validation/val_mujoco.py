# src/validation/val_mujoco.py
import os
import numpy as np
import torch

from src.utils.set_seed import set_seed

# Headless rendering default (set to "egl" if you have GPU EGL available)
os.environ.setdefault("MUJOCO_GL", "osmesa")

try:
    import gym, d4rl  # keep d4rl import to ensure gym IDs are registered if needed
except Exception as e:
    raise RuntimeError("gym is required for MuJoCo validation") from e

# Pull the same precomputed stats your dataset class uses
try:
    from src.envs_datasets.mujoco_dataset import PRECOMPUTED_DATASET_STATISTICS as _PRE_STATS
except Exception:
    _PRE_STATS = {}


@torch.no_grad()
def sample(
    model, x, block_size, steps, sample=False, top_k=None, actions=None,
    rtgs=None, timestep=None, mem_tokens=1, saved_context=None, hidden=None,
    memory_states=None, pos_offset=0
):
    """
    Framework-aligned sampling helper.
    Returns last-step 'logits' (treated as continuous action prediction),
    mem tokens/context/hidden/memory_states, and attn_map if exposed by the model.
    """
    model.eval()
    for _ in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:]
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:]

        if saved_context is not None:
            results = model(
                x_cond, actions, rtgs, None, timestep, *saved_context, mem_tokens=mem_tokens,
                hidden=hidden, memory_states=memory_states, pos_offset=pos_offset
            )
        else:
            results = model(
                x_cond, actions, rtgs, None, timestep, mem_tokens=mem_tokens,
                hidden=hidden, memory_states=memory_states, pos_offset=pos_offset
            )

        logits = results["logits"][:, -1, :]  # (B, A) at the last step
        memory = results.get("new_mems", None)
        mem_tokens = results.get("mem_tokens", None)
        hidden = results.get("hidden", None)
        attn_map = getattr(model, "attn_map", None)
        memory_states = results.get("memory_states", None)

    return logits, mem_tokens, memory, attn_map, hidden, memory_states


def _make_mujoco_env(env_name_str: str, seed: int):
    """
    Create a MuJoCo env consistent with the DT reference.
    Accepts short names ('hopper','halfcheetah','walker2d','reacher2d') or
    full Gym IDs ('Hopper-v3', etc.). For 'reacher2d' we instantiate your class.
    """
    name = env_name_str.lower()
    if name in ["hopper", "halfcheetah", "walker2d"]:
        gym_id = {"hopper": "Hopper-v3", "halfcheetah": "HalfCheetah-v3", "walker2d": "Walker2d-v3"}[name]
        env = gym.make(gym_id)
    elif name == "reacher2d":
        # Your custom env class
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
    else:
        # If already a gym id like "Hopper-v3"
        env = gym.make(env_name_str)

    try:
        env.seed(seed)  # gym<=0.21
    except Exception:
        pass
    print(f"Environment created: {env}")
    return env


def _set_env_normalization_scores(env, env_name_str: str):
    """
    Set ref_min_score and ref_max_score for environment normalization.
    Uses the same values as in train_rate_mujoco_ca.py
    """
    name = env_name_str.lower()
    
    if "halfcheetah" in name or "cheetah" in name:
        # HalfCheetah environments
        env.ref_max_score = 12135.0
        env.ref_min_score = -280.178953
    elif "walker" in name:
        # Walker2d environments  
        env.ref_max_score = 4592.3
        env.ref_min_score = 1.629008
    elif "hopper" in name:
        # Hopper environments
        env.ref_max_score = 3234.3
        env.ref_min_score = -20.272305
    else:
        # Default values for other environments
        env.ref_max_score = 1.0
        env.ref_min_score = 0.0
        
    return env


def _load_precomputed_stats(stats_key: str):
    """
    Load obs_mean/obs_std from your precomputed stats dict.
    """
    if stats_key not in _PRE_STATS:
        raise KeyError(
            f"No precomputed stats for key '{stats_key}'. "
            f"Available: {list(_PRE_STATS.keys())}"
        )
    stats = _PRE_STATS[stats_key]
    obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(stats["obs_std"], dtype=np.float32)
    obs_std = np.maximum(obs_std, 1e-6)
    return obs_mean, obs_std


@torch.no_grad()
def get_returns_MuJoCo(
    model,
    ret,                         # unscaled target return (float)
    seed,                        # int
    episode_timeout,             # int (e.g., 1000)
    context_length,              # K (int)
    device,                      # torch.device
    config,                      # dict with keys: ["dtype"], ["model_mode"], ["model"]["env_name"], ["model"]["act_dim"] (optional)
    use_argmax=False,            # unused for continuous; kept for signature parity
    create_video=False,          # if True, collects frames
    gif_path="mujoco_eval.gif",  # where to write GIF (if create_video)
    obs_mean=None,               # np.ndarray or None (explicit z-score mean)
    obs_std=None,                # np.ndarray or None (explicit z-score std)
    scale=None,                  # reward scale for RTG tracking. Defaults to 1000 for locomotion
    env=None,                    # optional pre-created environment to reuse
    normalize_obs: bool = True, # if True, use dataset-style normalization during eval
    stats_key: str = None,       # e.g. "halfcheetah-expert-v2" to match dataset distribution
):
    """
    Framework-aligned DT validation for MuJoCo vector tasks.

    Returns tuple consistent with your other validators:
        (episode_return, None, t, None, None, attn_map, frames_or_None)
    """
    # ---- dtype & seed ----
    dtype_map = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(config.get("dtype", "float32"), torch.float32)
    set_seed(seed)

    # ---- env ----
    should_close_env = False
    env_name = config["model"]["env_name"]  # e.g., "hopper", "HalfCheetah-v3", or "reacher2d"
    if env is None:
        env = _make_mujoco_env(env_name, seed)
        should_close_env = True
    else:
        try:
            env.seed(seed)
        except Exception:
            pass
    
    # Set normalization scores for return normalization (same as in train_rate_mujoco_ca.py)
    env = _set_env_normalization_scores(env, env_name)

    # ---- reward scaling (Decision Transformer convention) ----
    if scale is None:
        lname = str(env.spec.id if hasattr(env, "spec") and env.spec is not None else env_name).lower()
        if any(k in lname for k in ["hopper", "halfcheetah", "walker2d"]):
            scale = 1000.0
        else:
            scale = 1.0

    # ---- reset & dims ----
    state = env.reset()
    if isinstance(state, tuple):  # compat if env returns (obs, info)
        state = state[0]
    state = np.asarray(state, dtype=np.float32)

    state_dim = int(state.shape[0])
    if "model" in config and "act_dim" in config["model"]:
        act_dim = int(config["model"]["act_dim"])
    else:
        act_dim = int(env.action_space.shape[0])

    # ---- action bounds ----
    act_low = torch.as_tensor(getattr(env.action_space, "low", -np.ones(act_dim)), device=device, dtype=dtype)
    act_high = torch.as_tensor(getattr(env.action_space, "high", np.ones(act_dim)), device=device, dtype=dtype)

    # ---- normalize observations (dataset-style) ----
    if obs_mean is None or obs_std is None:
        if normalize_obs:
            key = config["data"]["path_to_dataset"].split("/")[-1].split(".")[0]
            obs_mean, obs_std = _load_precomputed_stats(key)

    if (obs_mean is not None) and (obs_std is not None):
        if len(obs_mean) != state_dim or len(obs_std) != state_dim:
            raise ValueError(
                f"Normalization stats dim mismatch: obs_dim={state_dim}, "
                f"mean_dim={len(obs_mean)}, std_dim={len(obs_std)}"
            )
        obs_mean_t = torch.as_tensor(obs_mean, device=device, dtype=dtype)
        obs_std_t = torch.as_tensor(np.maximum(obs_std, 1e-6), device=device, dtype=dtype)
        # print(f"Normalization stats loaded - mean shape: {obs_mean_t.shape}, std shape: {obs_std_t.shape}")
        # print(f"Mean values (first 5): {obs_mean_t[:5]}")
        # print(f"Std values (first 5): {obs_std_t[:5]}")
        def norm_obs(x: torch.Tensor) -> torch.Tensor:
            # print(f"Input tensor shape: {x.shape}")
            # print("Before normalization (per-feature stats):")
            # print(f"  min: {x.min(dim=-1)[0] if x.dim() > 1 else x.min()}")
            # print(f"  mean: {x.mean(dim=-1) if x.dim() > 1 else x.mean()}")
            # print(f"  max: {x.max(dim=-1)[0] if x.dim() > 1 else x.max()}")
            # print(f"  std: {x.std(dim=-1) if x.dim() > 1 else x.std()}")
            
            # Apply normalization: (x - dataset_mean) / dataset_std
            if x.dim() == 3:  # (batch, seq, features)
                mean_expanded = obs_mean_t.unsqueeze(0).unsqueeze(0)  # (1, 1, features)
                std_expanded = obs_std_t.unsqueeze(0).unsqueeze(0)    # (1, 1, features)
                normalized = (x - mean_expanded) / std_expanded
            elif x.dim() == 2:  # (seq, features) or (batch, features)
                normalized = (x - obs_mean_t) / obs_std_t
            else:
                normalized = (x - obs_mean_t) / obs_std_t
            
            # print("After normalization (per-feature stats):")
            # print(f"  min: {normalized.min(dim=-1)[0] if normalized.dim() > 1 else normalized.min()}")
            # print(f"  mean: {normalized.mean(dim=-1) if normalized.dim() > 1 else normalized.mean()}")
            # print(f"  max: {normalized.max(dim=-1)[0] if normalized.dim() > 1 else normalized.max()}")
            # print(f"  std: {normalized.std(dim=-1) if normalized.dim() > 1 else normalized.std()}")
            return normalized
    else:
        def norm_obs(x: torch.Tensor) -> torch.Tensor:
            return x

    # ---- histories (DT convention) ----
    states = torch.from_numpy(state).to(device=device, dtype=dtype).reshape(1, 1, state_dim)  # (1,S)
    actions = torch.zeros((0, act_dim), device=device, dtype=dtype)                         # (0,A)
    rewards = torch.zeros(0, device=device, dtype=dtype)                                    # (0,)
    target_return = torch.tensor(ret / scale, device=device, dtype=dtype).reshape(1, 1)     # (1,1) scaled
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)              # (1,1)

    # ---- memory/recurrent alignment ----
    is_lstm = hasattr(model, "backbone") and model.backbone in ["lstm", "gru"]
    mem_tokens = (model.mem_tokens.repeat(1, 1, 1).detach()
                  if hasattr(model, "mem_tokens") and model.mem_tokens is not None else None)
    saved_context = None
    hidden = model.reset_hidden(1, device) if is_lstm else None
    memory_states = model.init_memory(1, device) if config.get("model_mode", "") == "ELMUR" else None
    new_mem_tokens, new_context, new_memory_states = mem_tokens, saved_context, memory_states

    # ---- optional video ----
    frames = []
    if create_video:
        try:
            frame0 = env.render(mode="rgb_array")
            frames.append(frame0)
        except Exception:
            pass

    episode_return = 0.0
    attn_map = None

    for t in range(int(episode_timeout)):
        # pad slots (DT convention)
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device, dtype=dtype)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device, dtype=dtype)])

        # Sliding window (K)
        if not is_lstm and actions.shape[0] > context_length:
            slice_index = -1 if config["model_mode"] not in ['DT', 'DTXL'] else 1
            actions = actions[slice_index:] if slice_index == 1 else actions[slice_index:,:]
            states = states[:, slice_index:, :]
            target_return = target_return[:,slice_index:]
            timesteps = timesteps[:, slice_index:]
            if t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context
                if config["model_mode"] == "ELMUR":
                    memory_states = new_memory_states

        if is_lstm:
            states_to_pass = norm_obs(states[:, -1:, :])
            act_to_pass = None if t == 0 else actions[-1:].unsqueeze(0)
            rtg_to_pass = target_return[:, -1:].unsqueeze(-1)
            time_to_pass = timesteps[:, -1:]
        else:
            states_to_pass = norm_obs(states)
            act_to_pass = None if t == 0 else actions.unsqueeze(0)[:, 1:, :]
            rtg_to_pass = target_return.unsqueeze(-1)
            time_to_pass = timesteps
            if act_to_pass is not None and act_to_pass.shape[1] == 0:
                act_to_pass = None

        # For ELMUR we use the segment approach as during training
        if config["model_mode"] == "ELMUR":
            # Number of the current segment and position inside the segment
            segment_idx = t // context_length
            pos_in_segment = t % context_length
            # pos_offset corresponds to the beginning of the current segment
            # Get sequence format multiplier
            sequence_format = getattr(model, 'sequence_format', 'sra')
            multiplier = model.get_sequence_length_multiplier()
            pos_offset_val = segment_idx * context_length * multiplier
        else:
            window_len = min(context_length, t + 1)
            pos_offset_val = (t - window_len + 1) * 3

        # Model query
        logits, new_mem_tokens, new_context, attn_map, new_hidden, new_memory_states = sample(
            model=model,
            x=states_to_pass,
            block_size=context_length,
            steps=1,
            sample=True,
            actions=act_to_pass,
            rtgs=rtg_to_pass,
            timestep=time_to_pass,
            mem_tokens=mem_tokens,
            saved_context=saved_context,
            hidden=hidden,
            memory_states=memory_states,
            pos_offset=pos_offset_val,
        )

        if is_lstm:
            hidden = new_hidden

        # Continuous action prediction
        act = logits.reshape(-1)                      # (A,)
        act = torch.max(torch.min(act, act_high), act_low)  # clamp to bounds
        actions[-1] = act

        # Step env
        ns, reward, done, info = env.step(act.detach().cpu().numpy())
        if isinstance(ns, tuple):
            ns = ns[0]
        ns = np.asarray(ns, dtype=np.float32)

        # Update trajectories
        episode_return += float(reward)
        rewards[-1] = float(reward) / float(scale)

        cur_state = torch.from_numpy(ns).to(device=device, dtype=dtype).reshape(1, 1, -1)
        states = torch.cat([states, cur_state], dim=1)

        pred_return = target_return[0, -1] - (float(reward) / float(scale))
        target_return = torch.cat(
            [target_return, torch.tensor([[pred_return]], device=device, dtype=dtype)],
            dim=1
        )
        timesteps = torch.cat(
            [timesteps, torch.tensor([[t + 1]], device=device, dtype=torch.long)],
            dim=1
        )

        if create_video:
            try:
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)
            except Exception:
                pass

        if done:
            break

    # Optional GIF
    frames_out = None
    if create_video and len(frames) > 0:
        try:
            import imageio
            imageio.mimsave(gif_path, frames, fps=30)
            frames_out = frames
        except Exception:
            frames_out = None

    # Apply return normalization (same as in train_rate_mujoco_ca.py and eval_functions.py)
    # Formula: normalized_return = (raw_return - min_score) / (max_score - min_score) * 100
    # This converts raw environment returns to a 0-100 scale based on reference scores
    if hasattr(env, 'ref_min_score') and hasattr(env, 'ref_max_score'):
        episode_return = (episode_return - env.ref_min_score) / (env.ref_max_score - env.ref_min_score)
        episode_return = episode_return * 100.0
    else:
        raise ValueError(f"No ref_min_score or ref_max_score found for environment: {env_name}")
    
    if should_close_env:
        env.close()

    # Match your other validators' return signature:
    # (episode_return, None, t, None, None, attn_map, frames_or_None)
    return episode_return, t, frames_out
