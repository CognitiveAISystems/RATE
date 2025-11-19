#!/bin/bash

# Training script for Elastic-DT on ViZDoom Two Colors

# Default parameters
DATASET_DIR="${1:-/home/jovyan/echerepanov/RATE/data/ViZDoom_Two_Colors_150/}"
SEED="${2:-42}"

echo "Training Elastic-DT on ViZDoom Two Colors"
echo "Dataset: $DATASET_DIR"
echo "Seed: $SEED"

# * Initial parameters
# python Elastic-DT/scripts/train_edt_vizdoom.py \
#     --dataset_dir "$DATASET_DIR" \
#     --context_len 50 \
#     --n_blocks 4 \
#     --embed_dim 128 \
#     --n_heads 4 \
#     --dropout_p 0.1 \
#     --act_dim 5 \
#     --batch_size 128 \
#     --lr 1e-4 \
#     --wt_decay 1e-4 \
#     --warmup_steps 10000 \
#     --max_train_iters 500 \
#     --num_updates_per_iter 100 \
#     --model_save_iters 50 \
#     --rtg_scale 1000 \
#     --num_bin 60 \
#     --expectile 0.99 \
#     --exp_loss_weight 0.5 \
#     --ce_weight 0.001 \
#     --seed "$SEED" \
#     --device cuda \
#     --chk_pt_dir checkpoints/vizdoom/ \
#     --use_wandb \
#     --project_name EDT-ViZDoom

# * Small parameters to run test
python Elastic-DT/scripts/train_edt_vizdoom.py \
    --dataset_dir "$DATASET_DIR" \
    --context_len 150 \
    --n_blocks 2 \
    --embed_dim 32 \
    --n_heads 2 \
    --dropout_p 0.1 \
    --act_dim 5 \
    --batch_size 16 \
    --lr 1e-4 \
    --wt_decay 1e-4 \
    --warmup_steps 10000 \
    --max_train_iters 500 \
    --num_updates_per_iter 100 \
    --model_save_iters 2 \
    --rtg_scale 1 \
    --num_bin 60 \
    --expectile 0.99 \
    --exp_loss_weight 0.5 \
    --ce_weight 0.001 \
    --seed "$SEED" \
    --device cuda \
    --chk_pt_dir checkpoints/vizdoom/ \
    --use_wandb \
    --project_name EDT-ViZDoom

echo "Training complete!"

