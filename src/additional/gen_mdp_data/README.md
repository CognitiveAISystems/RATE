# MDP Data Generation for RATE/MATL

This directory contains scripts to train online RL policies on classic control MDP tasks, collect offline datasets, and analyze the resulting data for use with RATE/MATL models.

## Supported Environments

- **CartPole-v1**: Discrete action space, balance a pole on a cart
- **MountainCar-v0**: Discrete action space, drive a car up a hill
- **MountainCarContinuous-v0**: Continuous action space, drive a car up a hill
- **Acrobot-v1**: Discrete action space, swing up a two-link pendulum
- **Pendulum-v1**: Continuous action space, swing up and balance a pendulum

## Files Overview

- `train_online_rl.py`: Train state-of-the-art online RL policies
- `collect_mdp_datasets.py`: Collect offline datasets from trained policies
- `mdp_dataset.py`: PyTorch Dataset class for MDP data (compatible with RATE/MATL)
- `analyze_datasets.py`: Analyze and visualize dataset statistics
- `run_full_pipeline.py`: Run the complete pipeline (train → collect → analyze)
- `requirements.txt`: Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
cd src/additional/gen_mdp_data
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

Train policies, collect datasets, and analyze results for all environments:

```bash
python run_full_pipeline.py --env all
```

Or for a specific environment:

```bash
python run_full_pipeline.py --env CartPole-v1
```

### 3. Individual Steps

#### Train Online RL Policies

```bash
# Train best algorithm for each environment
python train_online_rl.py --env all --algorithm best

# Train specific algorithm on specific environment
python train_online_rl.py --env CartPole-v1 --algorithm PPO --timesteps 100000
```

#### Collect Offline Datasets

```bash
# Collect datasets from best trained models
python collect_mdp_datasets.py --env all --n-episodes 1000

# Collect from specific model
python collect_mdp_datasets.py --env CartPole-v1 --model-path models/CartPole-v1/PPO/best_model/best_model.zip --algorithm PPO
```

#### Analyze Datasets

```bash
# Analyze all datasets
python analyze_datasets.py --env all --plot

# Analyze specific environment
python analyze_datasets.py --env CartPole-v1 --data-base-dir data/MDP
```

## Algorithm Selection

The script automatically selects the best algorithm for each environment based on performance benchmarks:

- **CartPole-v1**: PPO
- **MountainCar-v0**: DQN
- **MountainCarContinuous-v0**: SAC
- **Acrobot-v1**: DQN
- **Pendulum-v1**: SAC

## Hyperparameters

Optimized hyperparameters are used for each environment-algorithm combination, based on RL baselines and research:

### CartPole-v1 (PPO)
- Learning rate: 3e-4
- Steps per update: 32
- Training timesteps: 100,000

### MountainCar-v0 (DQN)
- Learning rate: 4e-3
- Exploration decay: 0.2 fraction
- Training timesteps: 120,000

### MountainCarContinuous-v0 (SAC)
- Learning rate: 3e-4
- Buffer size: 300,000
- Training timesteps: 300,000

### Acrobot-v1 (DQN)
- Learning rate: 6.3e-4
- Target update interval: 250
- Training timesteps: 100,000

### Pendulum-v1 (SAC)
- Learning rate: 3e-4
- Batch size: 512
- Training timesteps: 300,000

## Dataset Format

The collected datasets follow the same format as existing RATE/MATL datasets:

```python
# Each episode file (train_data_X.npz) contains:
{
    'obs': np.array,      # Observations/states [T, state_dim]
    'action': np.array,   # Actions [T,] (discrete) or [T, action_dim] (continuous)
    'reward': np.array,   # Rewards [T,]
    'done': np.array      # Done flags [T,]
}
```

The `MDPDataset` class handles:
- Loading and preprocessing trajectory data
- Computing return-to-go (RTG) values
- Padding sequences to consistent length
- Converting to PyTorch tensors with appropriate dtypes

## Output Structure

```
src/additional/gen_mdp_data/
├── models/                          # Trained RL models
│   ├── CartPole-v1/PPO/
│   ├── MountainCar-v0/DQN/
│   └── ...
├── analysis/                        # Analysis results
│   ├── analysis_summary.json
│   ├── dataset_analysis.png
│   └── dataset_analysis.pdf
├── training_summary.json            # Training results summary
└── dataset_collection_summary.json  # Collection results summary

data/MDP/                            # Dataset files
├── CartPole-v1/
│   ├── train_data_0.npz
│   ├── train_data_1.npz
│   ├── ...
│   └── dataset_metadata.json
├── MountainCar-v0/
└── ...
```

## Integration with RATE/MATL

To use the collected MDP datasets with RATE/MATL:

1. **Add dataset to envs_datasets**: Create a new dataset class in `src/envs_datasets/` that inherits from or follows the pattern of existing datasets.

2. **Update trainer configuration**: Add MDP environment configurations to your training config files.

3. **Use the MDPDataset class**: The provided `MDPDataset` class is already compatible with the RATE/MATL training pipeline.

Example usage:

```python
from src.additional.gen_mdp_data.mdp_dataset import create_mdp_dataloader

# Create dataloader for CartPole
dataloader = create_mdp_dataloader(
    data_dir="data/MDP/CartPole-v1",
    env_name="CartPole-v1", 
    batch_size=32,
    max_length=500
)

# Use with RATE/MATL trainer
trainer.train(dataloader)
```

## Performance Expectations

Expected performance ranges for trained policies:

- **CartPole-v1**: 450-500 (max 500)
- **MountainCar-v0**: -110 to -90 (solved at -110)
- **MountainCarContinuous-v0**: 85-95 (solved at 90)
- **Acrobot-v1**: -80 to -100 (solved at -100)
- **Pendulum-v1**: -150 to -200 (max ~-123)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `n_envs` parameter or use CPU training
2. **Slow training**: Increase `n_envs` for parallel environment sampling
3. **Poor performance**: Check hyperparameters or increase training timesteps
4. **Missing models**: Ensure training completed successfully before collection

### Memory Usage

- Training: ~1-2GB RAM per environment (with 4 parallel envs)
- Collection: ~500MB-1GB RAM depending on episode length
- Analysis: ~200-500MB RAM for loading datasets

## Citation

If you use this code in your research, please cite the original RATE/MATL papers and acknowledge the use of Stable-Baselines3 for policy training.
