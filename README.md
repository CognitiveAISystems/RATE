<h1 align="center">Recurrent Action Transformer with Memory (RATE)</h1>
<h3 align="center">Offline RL agent for solving memory-intensive tasks</h3>

<div align="center">
    <a href="https://arxiv.org/abs/2306.09459">
        <img src="https://img.shields.io/badge/arXiv-2306.09459-b31b1b.svg"/>
    </a>
    <a href="https://sites.google.com/view/rate-model/">
        <img src="https://img.shields.io/badge/Website-Project_Page-blue.svg"/>
    </a>
    <a href="https://github.com/CognitiveAISystems/RATE">
        <img src="https://img.shields.io/badge/GitHub-RATE-green.svg"/>
    </a>
    <a href="https://github.com/CognitiveAISystems/RATE/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
    </a>
</div>

## Overview

RATE (Recurrent Action Transformer with Memory) is a novel offline RL agent designed specifically for solving memory-intensive tasks. The model combines transformer architecture with memory mechanisms to effectively handle long-term dependencies in sequential decision-making tasks. 

Update: The RATE training framework also includes the [ELMUR (External Layer Memory with Update/Rewrite)](offline_rl_baselines/ELMUR) model - a novel memory-augmented transformer architecture designed for long-horizon RL tasks.

## Implementation Status

<table>
<tr>
<td valign="top">

### 🌐 Environments
| Status | Environment |
|:------:|-------------|
| ✅ | T-Maze |
| ✅ | ViZDoom-Two-Colors |
| ✅ | Minigrid-Memory |
| ✅ | Memory-Maze |
| ✅ | POPGym |
| 🔄 | Action-Associative-Retrieval |
| ✅ | Atari |
| ✅ | MuJoCo |

</td>
<td valign="top">

### 🤖 Baselines
| Status | Algorithm |
|:------:|-----------|
| ✅ | RATE |
| ✅ | ELMUR |
| ✅ | DT |
| ✅ | RMT |
| ✅ | TrXL |
| ✅ | BC (MLP / LSTM) |
| ✅ | CQL (MLP / LSTM) |
| ✅ | IQL |
| ✅ | [DLSTM](https://github.com/max7born/decision-lstm) / DGRU |
| ✅ | [DMamba](https://github.com/Toshihiro-Ota/decision-mamba) |
| ✅ | [LSDT](https://github.com/WangJinCheng1998/LONG-SHORT-DECISION-TRANSFORMER) |
| ✅ | Diffusion Policy |

</td>
</tr>
</table>

## Requirements

```bash
# Install main dependencies
pip install -e .

# Install additional dependencies for ViZDoom-Two-Colors
pip install -r requirements/requirements_vizdoom_two_colors.txt

# Install additional dependencies for Minigrid-Memory
pip install -r requirements/requirements_minigrid_memory.txt

# Install additional dependencies for Memory-Maze
pip install -r requirements/requirements_memory_maze.txt

# Install additional dependencies for POPGym
pip install -r requirements/requirements_popgym.txt
```

## Memory-intensive tasks

### T-Maze
Configuration file: [run_experiments/TMaze.sh](run_experiments/TMaze.sh)
- Environment: Simple memory task with T-shaped maze
- Supported baselines: RATE, DT, RMT, TrXL, LSDT, BC-MLP, BC-LSTM, CQL-MLP, CQL-LSTM, DLSTM, DGRU, DMamba

### ViZDoom-Two-Colors
Configuration file: [run_experiments/ViZDoom_Two_colors.sh](run_experiments/ViZDoom_Two_colors.sh)
- Environment: First-person shooter with color-based memory task
- Supported baselines: RATE, DT, RMT, TrXL, LSDT, BC-MLP, BC-LSTM, CQL-MLP, CQL-LSTM, DLSTM, DGRU, DMamba

### Minigrid-Memory
Configuration file: [run_experiments/Minigrid_Memory.sh](run_experiments/Minigrid_Memory.sh)
- Environment: Grid-based memory task
- Supported baselines: RATE, DT, RMT, TrXL, LSDT, BC-MLP, BC-LSTM, CQL-MLP, CQL-LSTM, DLSTM, DGRU, DMamba

### Memory-Maze
Configuration file: [run_experiments/Memory_Maze.sh](run_experiments/Memory_Maze.sh)
- Environment: Complex maze navigation with memory requirements
- Supported baselines: RATE, DT, RMT, TrXL

### POPGym
Configuration file: [run_experiments/POPGym.sh](run_experiments/POPGym.sh)
- Environment: Suite of 48 partially observable environments
- Supported baselines: RATE, DT, BC-MLP, BC-LSTM
- Available tasks: See [POPGym documentation](https://github.com/proroklab/popgym) for details, and the table below for mapping indices to the task name

For each environment:
1. Open the corresponding configuration file
2. Select the desired baseline model
3. Adjust hyperparameters if needed
4. Run the training script

### POPGym and MIKASA-Robo dataset descriptions

| Index | Environment |
|-------|-------------|
| 0 | popgym-AutoencodeEasy-v0 |
| 1 | popgym-AutoencodeHard-v0 |
| 2 | popgym-AutoencodeMedium-v0 |
| 3 | popgym-BattleshipEasy-v0 |
| 4 | popgym-BattleshipHard-v0 |
| 5 | popgym-BattleshipMedium-v0 |
| 6 | popgym-ConcentrationEasy-v0 |
| 7 | popgym-ConcentrationHard-v0 |
| 8 | popgym-ConcentrationMedium-v0 |
| 9 | popgym-CountRecallEasy-v0 |
| 10 | popgym-CountRecallHard-v0 |
| 11 | popgym-CountRecallMedium-v0 |
| 12 | popgym-HigherLowerEasy-v0 |
| 13 | popgym-HigherLowerHard-v0 |
| 14 | popgym-HigherLowerMedium-v0 |
| 15 | popgym-LabyrinthEscapeEasy-v0 |
| 16 | popgym-LabyrinthEscapeHard-v0 |
| 17 | popgym-LabyrinthEscapeMedium-v0 |
| 18 | popgym-LabyrinthExploreEasy-v0 |
| 19 | popgym-LabyrinthExploreHard-v0 |
| 20 | popgym-LabyrinthExploreMedium-v0 |
| 21 | popgym-MineSweeperEasy-v0 |
| 22 | popgym-MineSweeperHard-v0 |
| 23 | popgym-MineSweeperMedium-v0 |
| 24 | popgym-MultiarmedBanditEasy-v0 |
| 25 | popgym-MultiarmedBanditHard-v0 |
| 26 | popgym-MultiarmedBanditMedium-v0 |
| 27 | popgym-NoisyPositionOnlyCartPoleEasy-v0 |
| 28 | popgym-NoisyPositionOnlyCartPoleHard-v0 |
| 29 | popgym-NoisyPositionOnlyCartPoleMedium-v0 |
| 30 | popgym-NoisyPositionOnlyPendulumEasy-v0 |
| 31 | popgym-NoisyPositionOnlyPendulumHard-v0 |
| 32 | popgym-NoisyPositionOnlyPendulumMedium-v0 |
| 33 | popgym-PositionOnlyCartPoleEasy-v0 |
| 34 | popgym-PositionOnlyCartPoleHard-v0 |
| 35 | popgym-PositionOnlyCartPoleMedium-v0 |
| 36 | popgym-PositionOnlyPendulumEasy-v0 |
| 37 | popgym-PositionOnlyPendulumHard-v0 |
| 38 | popgym-PositionOnlyPendulumMedium-v0 |
| 39 | popgym-RepeatFirstEasy-v0 |
| 40 | popgym-RepeatFirstHard-v0 |
| 41 | popgym-RepeatFirstMedium-v0 |
| 42 | popgym-RepeatPreviousEasy-v0 |
| 43 | popgym-RepeatPreviousHard-v0 |
| 44 | popgym-RepeatPreviousMedium-v0 |
| 45 | popgym-VelocityOnlyCartpoleEasy-v0 |
| 46 | popgym-VelocityOnlyCartpoleHard-v0 |
| 47 | popgym-VelocityOnlyCartpoleMedium-v0 |

## Data collection
Experiments with T-Maze can be run without data collection. For other experiments, you need to collect data. Here we provide scripts for data collection for ViZDoom-Two-Colors and Minigrid-Memory.

```bash
# Create data for ViZDoom-Two-Colors:
python3 src/additional/gen_vizdoom_data/gen_vizdoom_data.py

# Create data for Minigrid-Memory:
# --random False -> one corridor length with one grid size in data (default in paper)
# --random True -> different corridor lengths with one grid size from min to max
python3 src/additional/gen_minigrid_memory_data/gen_minigrid_memory_data.py --random False

# Download data for POPGym (247 Mb)
mkdir -p data
wget https://huggingface.co/datasets/avanturist/popgym-datasets-48-tasks/resolve/main/POPGym.zip -O data/POPGym.zip
unzip -q data/POPGym.zip -d data/

# Memory Maze
echo Data can be collected with https://github.com/NM512/dreamerv3-torch
```


## Classic Baselines
### Atari

#### Installation

Dependencies can be installed with the following command:

```bash
cd Atari_MuJoCo/Atari
conda env create -f conda_env.yml
```

#### Downloading datasets


Create a directory for the dataset and load the dataset using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

In the `wandb_config.yaml` in the main directory add the following lines to specify the directory with Atari data:

```python
atari:
  data: '/path/to/atari/data/'
```

#### Example usage

```python
python3 Atari/train_rate_atari.py --game Breakout --num_mem_tokens 15 --mem_len 360 --n_head_ca 1 --mrv_act 'relu' --skip_dec_ffn --seed 123
python3 Atari/train_rate_atari.py --game Qbert --num_mem_tokens 15 --mem_len 360 --n_head_ca 1 --mrv_act 'relu' --skip_dec_ffn --seed 123
python3 Atari/train_rate_atari.py --game Seaquest --num_mem_tokens 15 --mem_len 360 --n_head_ca 1 --mrv_act 'relu' --skip_dec_ffn --seed 123
python3 Atari/train_rate_atari.py --game Pong --num_mem_tokens 15 --mem_len 360 --n_head_ca 1 --mrv_act 'leaky_relu' --skip_dec_ffn --seed 123
```

### MuJoCo

#### Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```bash
cd Atari_MuJoCo/MuJoCo
conda env create -f conda_env.yml
```

### Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

In the `wandb_config.yaml` in the main directory add the following lines:

```python
mujoco:
  data_dir_prefix: '/path/to/mujoco/data/'
```

#### Example usage

```python
python3 MuJoCo/train_rate_mujoco_ca.py --env_id 0 --number_of_segments 3 --segment_length 20 --num_mem_tokens 5 --n_head_ca 1 --mrv_act 'relu' --skip_dec_ffn --seed 123
```

Where `env_id` - id of MuJoCo task:

1. `env_id` - MuJoCo environment id:
    - 0 → `halfcheetah-medium`
    - 1 → `halfcheetah-medium-replay`
    - 2 → `halfcheetah-expert`
    - 3 → `walker2d-medium`
    - 4 → `walker2d-medium-replay`
    - 5 → `walker2d-expert`
    - 6 → `hopper-medium`
    - 7 → `hopper-medium-replay`
    - 8 → `hopper-expert`
    - 9 → `halfcheetah-medium-expert`
    - 10 → `walker2d-medium-expert`
    - 11 → `hopper-medium-expert`

## Citation
If you find our work useful, please cite our paper:
```
@misc{cherepanov2024recurrentactiontransformermemory,
      title={Recurrent Action Transformer with Memory}, 
      author={Egor Cherepanov and Alexey Staroverov and Dmitry Yudin and Alexey K. Kovalev and Aleksandr I. Panov},
      year={2024},
      eprint={2306.09459},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2306.09459}, 
}
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.