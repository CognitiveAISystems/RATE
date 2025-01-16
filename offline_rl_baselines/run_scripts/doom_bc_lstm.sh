#!/bin/bash

for seed in {1..3};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_doom_cql_bc_lstm.py \
    --seed $seed \
    --exp-name 'doom' \
    --loss-mode=bc \
    --num-epochs=1000
done