#!/bin/bash

for seed in {1..6};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_doom_cql_bc_mlp.py \
    --seed $seed \
    --exp-name 'doom_mlp' \
    --loss-mode=bc \
    --num-epochs=500
done