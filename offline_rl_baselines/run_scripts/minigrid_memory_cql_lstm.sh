#!/bin/bash

for seed in {1..3};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_minigrid_memory_cql_bc_lstm.py \
    --seed $seed \
    --exp-name 'minigrid_memory' \
    --loss-mode=cql \
    --num-epochs=1000
done