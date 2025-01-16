# * DOOM
python3 offline_rl_baselines/train_doom_cql_bc_lstm.py --seed 1 --exp-name 'doom' --loss-mode=cql --num-epochs=1000

# * TMaze
python3 offline_rl_baselines/train_tmaze_cql_bc_lstm.py --seed 1 --exp-name 'tmaze' --loss-mode=cql --num-epochs=1000 --segments=3 --context-length=30

# * Memory Maze
python3 offline_rl_baselines/train_memory_maze_cql_bc_lstm.py --seed 1 --exp-name 'memory_maze' --loss-mode=cql --num-epochs=1000

# * Minigrid Memory
python3 offline_rl_baselines/train_minigrid_memory_cql_bc_lstm.py --seed 1 --exp-name 'minigrid_memory' --loss-mode=cql --num-epochs=1000