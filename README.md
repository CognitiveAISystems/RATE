# Recurrent Action Transformer with Memory (RATE) 

## Important information

> [!important]
> ⚠️ **Repository Status: Major Refactoring in Progress**  
> This repository is currently undergoing significant architectural improvements. While some features are temporarily limited, you can access the complete previous version in the [`rate-old` branch](https://github.com/CognitiveAISystems/RATE/tree/rate-old).


Refactoring status (what is available now):
- ✅ Clean universal model and traner code
- ✅ T-Maze
- ✅ ViZDoom-Two-Colors
- ✅ Minigrid-Memory
- ✅ Memory-Maze
- 🔄 POPGym
- 🔄 [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)
- 🔄 Action-Associative-Retrieval
- 🔄 Atari
- 🔄 MuJoCo

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
TBD

# Install additional dependencies for MIKASA-Robo
pip install mikasa_robo_suite
```

## Run experiments

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

```

## Data collection
Experiments with T-Maze can be run without data collection. For other experiments, you need to collect data. Here we provide scripts for data collection for ViZDoom-Two-Colors and Minigrid-Memory.

```bash
# Create data for ViZDoom-Two-Colors:
python3 src/additional/gen_vizdoom_data/gen_vizdoom_data.py

# Create data for Minigrid-Memory:
python3 src/additional/gen_minigrid_memory_data/gen_minigrid_memory_data.py
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

MIT
