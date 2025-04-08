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

> [!important]
> ⚠️ **Repository Status: Major Refactoring in Progress**  
> The code for Atari and MuJoCo is in the process of being refactored and is available in the previous version in the [`rate-old` branch](https://github.com/CognitiveAISystems/RATE/tree/rate-old).


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
| ✅ | [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo) |
| 🔄 | Action-Associative-Retrieval |
| 🔄 | Atari |
| 🔄 | MuJoCo |

</td>
<td valign="top">

### 🤖 Baselines
| Status | Algorithm |
|:------:|-----------|
| ✅ | RATE |
| ✅ | DT |
| ✅ | RMT |
| ✅ | TrXL |
| ✅ | BC |
| ✅ | CQL |
| ✅ | IQL |
| 🔄 | Diffusion Policy |
| 🔄 | Trajectory Transformer |

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

# Install additional dependencies for PopGym [TBD]
pip install -r requirements/requirements_popgym.txt

# Install additional dependencies for MIKASA-Robo
pip install mikasa_robo_suite
```

## Run experiments

### Running commands

```bash
# T-Maze
bash run_experiments/TMaze/run_rate_tmaze.sh
bash run_experiments/TMaze/run_dt_tmaze.sh
bash run_experiments/TMaze/run_rmt_tmaze.sh
bash run_experiments/TMaze/run_trxl_tmaze.sh

# ViZDoom-Two-Colors
bash run_experiments/ViZDoom_Two_Colors/run_rate_vizdoom_two_colors.sh
bash run_experiments/ViZDoom_Two_Colors/run_dt_vizdoom_two_colors.sh
bash run_experiments/ViZDoom_Two_Colors/run_rmt_vizdoom_two_colors.sh
bash run_experiments/ViZDoom_Two_Colors/run_trxl_vizdoom_two_colors.sh

# Minigrid-Memory
bash run_experiments/Minigrid_Memory/run_rate_minigrid_memory.sh
bash run_experiments/Minigrid_Memory/run_dt_minigrid_memory.sh
bash run_experiments/Minigrid_Memory/run_rmt_minigrid_memory.sh
bash run_experiments/Minigrid_Memory/run_trxl_minigrid_memory.sh

# Memory-Maze
bash run_experiments/Memory_Maze/run_rate_memory_maze.sh
bash run_experiments/Memory_Maze/run_dt_memory_maze.sh
bash run_experiments/Memory_Maze/run_rmt_memory_maze.sh
bash run_experiments/Memory_Maze/run_trxl_memory_maze.sh

# POPGym
# [index] value in [0, 47]
bash run_experiments/POPGym/DT/[index].sh
bash run_experiments/POPGym/RATE/[index].sh

# MIKASA-Robo
# [index] value in [0, 31]
bash run_experiments/MIKASA_Robo/RATE/K_Tdiv3/[index].sh
bash run_experiments/MIKASA_Robo/DT/K_Tdiv3/[index].sh
bash run_experiments/MIKASA_Robo/BC_MLP/[index].sh
bash run_experiments/MIKASA_Robo/CQL_MLP/[index].sh
bash run_experiments/MIKASA_Robo/IQL_MLP/[index].sh

```

### POPGym and MIKASA-Robo dataset descriptions

<table>
<tr>
<td valign="top">

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

</td>
<td valign="top">

| Index | Environment |
|-------|-------------|
| 0 | ShellGameTouch-v0 |
| 1 | ShellGamePush-v0 |
| 2 | ShellGamePick-v0 |
| 3 | InterceptSlow-v0 |
| 4 | InterceptMedium-v0 |
| 5 | InterceptFast-v0 |
| 6 | InterceptGrabSlow-v0 |
| 7 | InterceptGrabMedium-v0 |
| 8 | InterceptGrabFast-v0 |
| 9 | RotateLenientPos-v0 |
| 10 | RotateLenientPosNeg-v0 |
| 11 | RotateStrictPos-v0 |
| 12 | RotateStrictPosNeg-v0 |
| 13 | TakeItBack-v0 |
| 14 | RememberColor3-v0 |
| 15 | RememberColor5-v0 |
| 16 | RememberColor9-v0 |
| 17 | RememberShape3-v0 |
| 18 | RememberShape5-v0 |
| 19 | RememberShape9-v0 |
| 20 | RememberShapeAndColor3x2-v0 |
| 21 | RememberShapeAndColor3x3-v0 |
| 22 | RememberShapeAndColor5x3-v0 |
| 23 | BunchOfColors3-v0 |
| 24 | BunchOfColors5-v0 |
| 25 | BunchOfColors7-v0 |
| 26 | SeqOfColors3-v0 |
| 27 | SeqOfColors5-v0 |
| 28 | SeqOfColors7-v0 |
| 29 | ChainOfColors3-v0 |
| 30 | ChainOfColors5-v0 |
| 31 | ChainOfColors7-v0 |

</td>
</tr>
</table>

## Data collection
Experiments with T-Maze can be run without data collection. For other experiments, you need to collect data. Here we provide scripts for data collection for ViZDoom-Two-Colors and Minigrid-Memory.

```bash
# Create data for ViZDoom-Two-Colors:
python3 src/additional/gen_vizdoom_data/gen_vizdoom_data.py

# Create data for Minigrid-Memory:
python3 src/additional/gen_minigrid_memory_data/gen_minigrid_memory_data.py

# Download data for POPGym (247 Mb)
mkdir -p data
wget https://huggingface.co/datasets/avanturist/popgym-datasets-48-tasks/resolve/main/POPGym.zip -O data/POPGym.zip
unzip -q data/POPGym.zip -d data/

# Download data for MIKASA-Robo
echo Follow instructions from README.md in https://github.com/CognitiveAISystems/MIKASA-Robo
```

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