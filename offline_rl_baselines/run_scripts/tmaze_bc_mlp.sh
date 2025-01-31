#!/bin/bash

for seed in {1..10};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_tmaze_cql_bc_mlp.py \
    --seed $seed \
    --exp-name 'tmaze_mlp' \
    --loss-mode=bc \
    --num-epochs=100 \
    --segments=3 \
    --context-length=3
done


for seed in {1..10};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_tmaze_cql_bc_mlp.py \
    --seed $seed \
    --exp-name 'tmaze_mlp' \
    --loss-mode=bc \
    --num-epochs=100 \
    --segments=3 \
    --context-length=10
done


for seed in {1..10};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_tmaze_cql_bc_mlp.py \
    --seed $seed \
    --exp-name 'tmaze_mlp' \
    --loss-mode=bc \
    --num-epochs=100 \
    --segments=3 \
    --context-length=30
done


for seed in {1..10};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_tmaze_cql_bc_mlp.py \
    --seed $seed \
    --exp-name 'tmaze_mlp' \
    --loss-mode=bc \
    --num-epochs=100 \
    --segments=3 \
    --context-length=90
done