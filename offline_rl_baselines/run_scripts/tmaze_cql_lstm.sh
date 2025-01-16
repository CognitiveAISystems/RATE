#!/bin/bash

for seed in {1..10};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_tmaze_cql_bc_lstm.py \
    --seed $seed \
    --exp-name 'tmaze' \
    --loss-mode=cql \
    --num-epochs=500 \
    --segments=3 \
    --context-length=3
done


for seed in {1..10};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_tmaze_cql_bc_lstm.py \
    --seed $seed \
    --exp-name 'tmaze' \
    --loss-mode=cql \
    --num-epochs=500 \
    --segments=3 \
    --context-length=10
done


for seed in {1..10};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_tmaze_cql_bc_lstm.py \
    --seed $seed \
    --exp-name 'tmaze' \
    --loss-mode=cql \
    --num-epochs=500 \
    --segments=3 \
    --context-length=30
done


for seed in {1..10};
do
    echo "Running experiment with seed $seed"
    python3 offline_rl_baselines/train_tmaze_cql_bc_lstm.py \
    --seed $seed \
    --exp-name 'tmaze' \
    --loss-mode=cql \
    --num-epochs=500 \
    --segments=3 \
    --context-length=90
done