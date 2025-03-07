import ray
import torch
from ray.tune.registry import register_env
import popgym
from popgym import wrappers
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.core.env import POPGymEnv
import numpy as np
from dataclasses import dataclass
import tyro


env_names_dict = {
    0: ["popgym-AutoencodeEasy-v0", popgym.envs.autoencode.AutoencodeEasy],
    1: ["popgym-RepeatPreviousEasy-v0", popgym.envs.repeat_previous.RepeatPreviousEasy],
    2: ["popgym-RepeatFirstEasy-v0", popgym.envs.repeat_first.RepeatFirstEasy],
    3: ["popgym-CountRecallEasy-v0", popgym.envs.count_recall.CountRecallEasy],
    4: ["popgym-PositionOnlyCartPoleEasy-v0", popgym.envs.position_only_cartpole.PositionOnlyCartPoleEasy],
    5: ["popgym-PositionOnlyPendulumEasy-v0", popgym.envs.position_only_pendulum.PositionOnlyPendulumEasy],
    6: ["popgym-VelocityOnlyCartpoleEasy-v0", popgym.envs.velocity_only_cartpole.VelocityOnlyCartPoleEasy],
    7: ["popgym-NoisyPositionOnlyCartPoleEasy-v0", popgym.envs.noisy_position_only_cartpole.NoisyPositionOnlyCartPoleEasy],
    8: ["popgym-NoisyPositionOnlyPendulumEasy-v0", popgym.envs.noisy_position_only_pendulum.NoisyPositionOnlyPendulumEasy],
    9: ["popgym-MultiarmedBanditEasy-v0", popgym.envs.multiarmed_bandit.MultiarmedBanditEasy],
    10: ["popgym-HigherLowerEasy-v0", popgym.envs.higher_lower.HigherLowerEasy],
    11: ["popgym-BattleshipEasy-v0", popgym.envs.battleship.BattleshipEasy],
    12: ["popgym-ConcentrationEasy-v0", popgym.envs.concentration.ConcentrationEasy],
    13: ["popgym-MineSweeperEasy-v0", popgym.envs.minesweeper.MineSweeperEasy],
    14: ["popgym-LabyrinthExploreEasy-v0", popgym.envs.labyrinth_explore.LabyrinthExploreEasy],
    15: ["popgym-LabyrinthEscapeEasy-v0", popgym.envs.labyrinth_escape.LabyrinthEscapeEasy],
    16: ["popgym-AutoencodeMedium-v0", popgym.envs.autoencode.AutoencodeMedium],
    17: ["popgym-RepeatPreviousMedium-v0", popgym.envs.repeat_previous.RepeatPreviousMedium],
    18: ["popgym-RepeatFirstMedium-v0", popgym.envs.repeat_first.RepeatFirstMedium],
    19: ["popgym-CountRecallMedium-v0", popgym.envs.count_recall.CountRecallMedium],
    20: ["popgym-PositionOnlyCartPoleMedium-v0", popgym.envs.position_only_cartpole.PositionOnlyCartPoleMedium],
    21: ["popgym-PositionOnlyPendulumMedium-v0", popgym.envs.position_only_pendulum.PositionOnlyPendulumMedium],
    22: ["popgym-VelocityOnlyCartpoleMedium-v0", popgym.envs.velocity_only_cartpole.VelocityOnlyCartPoleMedium],
    23: ["popgym-NoisyPositionOnlyCartPoleMedium-v0", popgym.envs.noisy_position_only_cartpole.NoisyPositionOnlyCartPoleMedium],
    24: ["popgym-NoisyPositionOnlyPendulumMedium-v0", popgym.envs.noisy_position_only_pendulum.NoisyPositionOnlyPendulumMedium],
    25: ["popgym-MultiarmedBanditMedium-v0", popgym.envs.multiarmed_bandit.MultiarmedBanditMedium],
    26: ["popgym-HigherLowerMedium-v0", popgym.envs.higher_lower.HigherLowerMedium],
    27: ["popgym-BattleshipMedium-v0", popgym.envs.battleship.BattleshipMedium],
    28: ["popgym-ConcentrationMedium-v0", popgym.envs.concentration.ConcentrationMedium],
    29: ["popgym-MineSweeperMedium-v0", popgym.envs.minesweeper.MineSweeperMedium],
    30: ["popgym-LabyrinthExploreMedium-v0", popgym.envs.labyrinth_explore.LabyrinthExploreMedium],
    31: ["popgym-LabyrinthEscapeMedium-v0", popgym.envs.labyrinth_escape.LabyrinthEscapeMedium],
    32: ["popgym-AutoencodeHard-v0", popgym.envs.autoencode.AutoencodeHard],
    33: ["popgym-RepeatPreviousHard-v0", popgym.envs.repeat_previous.RepeatPreviousHard],
    34: ["popgym-RepeatFirstHard-v0", popgym.envs.repeat_first.RepeatFirstHard],
    35: ["popgym-CountRecallHard-v0", popgym.envs.count_recall.CountRecallHard],
    36: ["popgym-PositionOnlyCartPoleHard-v0", popgym.envs.position_only_cartpole.PositionOnlyCartPoleHard],
    37: ["popgym-PositionOnlyPendulumHard-v0", popgym.envs.position_only_pendulum.PositionOnlyPendulumHard],
    38: ["popgym-VelocityOnlyCartpoleHard-v0", popgym.envs.velocity_only_cartpole.VelocityOnlyCartPoleHard],
    39: ["popgym-NoisyPositionOnlyCartPoleHard-v0", popgym.envs.noisy_position_only_cartpole.NoisyPositionOnlyCartPoleHard],
    40: ["popgym-NoisyPositionOnlyPendulumHard-v0", popgym.envs.noisy_position_only_pendulum.NoisyPositionOnlyPendulumHard],
    41: ["popgym-MultiarmedBanditHard-v0", popgym.envs.multiarmed_bandit.MultiarmedBanditHard],
    42: ["popgym-HigherLowerHard-v0", popgym.envs.higher_lower.HigherLowerHard],
    43: ["popgym-BattleshipHard-v0", popgym.envs.battleship.BattleshipHard],
    44: ["popgym-ConcentrationHard-v0", popgym.envs.concentration.ConcentrationHard],
    45: ["popgym-MineSweeperHard-v0", popgym.envs.minesweeper.MineSweeperHard],
    46: ["popgym-LabyrinthExploreHard-v0", popgym.envs.labyrinth_explore.LabyrinthExploreHard],
    47: ["popgym-LabyrinthEscapeHard-v0", popgym.envs.labyrinth_escape.LabyrinthEscapeHard],
}


def collect_checkpoints_for_group(group):
    for env_idx in group:
        train_popgym_by_env_idx(env_idx)

def wrap(env: POPGymEnv) -> POPGymEnv:
    return wrappers.Antialias(wrappers.PreviousAction(env))

def train_popgym_by_env_idx(env_idx):
    env_name = env_names_dict[env_idx][0]
    env_class = env_names_dict[env_idx][1]
    register_env(env_name, lambda x: wrap(env_class()))

    # Configuration parameters
    bptt_cutoff = 1024
    num_workers = 4
    num_envs_per_worker = 16
    gpu_per_worker = 0.25
    max_steps = 15_000_000

    # Hidden sizes
    h = 128  # Hidden size of linear layers
    h_memory = 256  # Hidden size of memory

    config = {
        "env": env_name,
        "framework": "torch",
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "num_gpus": gpu_per_worker,
        
        # PPO specific parameters from the table
        "gamma": 0.99,                    # Decay factor
        "vf_loss_coeff": 1.0,            # Value function loss coefficient
        "entropy_coeff": 0.0,            # Entropy loss coefficient
        "lr": 5e-5,                      # Learning rate
        "num_sgd_iter": 30,              # Number of SGD iterations
        "train_batch_size": 65536,       # Batch size
        "sgd_minibatch_size": 8192,      # Minibatch size
        "clip_param": 0.3,               # PPO clipping parameter
        "vf_clip_param": 0.3,            # Value function clipping
        "kl_target": 0.01,               # KL target
        "kl_coeff": 0.2,                 # KL coefficient
        
        # Other parameters
        "horizon": 1024,                 # Maximum Episode Length
        "batch_mode": "complete_episodes",
        "rollout_fragment_length": bptt_cutoff,
        
        "model": {
            "max_seq_len": bptt_cutoff,
            "custom_model": GRU,
            "custom_model_config": {
                "preprocessor_input_size": h,
                "preprocessor": torch.nn.Sequential(
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "preprocessor_output_size": h,
                "hidden_size": h_memory,
                "postprocessor": torch.nn.Identity(),
                "actor": torch.nn.Sequential(
                    torch.nn.Linear(h_memory, h),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "critic": torch.nn.Sequential(
                    torch.nn.Linear(h_memory, h),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "postprocessor_output_size": h,
            },
        },
    }

    # Stop condition
    stop = {"timesteps_total": max_steps}

    result = ray.tune.run(
        "PPO",
        config=config,
        stop=stop,
        verbose=1,
        local_dir="popgym/results",
        checkpoint_freq=10,
        keep_checkpoints_num=1,
    )

@dataclass
class TrainingArgs:
    group: int = 1  # Group number (1-4) to train

if __name__ == "__main__":
    args = tyro.cli(TrainingArgs)
    
    group_1 = np.arange(0, 12)
    group_2 = np.arange(12, 24)
    group_3 = np.arange(24, 36)
    group_4 = np.arange(36, 48)
    
    groups = {
        1: group_1,
        2: group_2,
        3: group_3,
        4: group_4
    }
    
    if args.group not in groups:
        raise ValueError(f"Group must be between 1 and 4, got {args.group}")
        
    print(f"Collecting checkpoints for group {args.group}")

    collect_checkpoints_for_group(groups[args.group])