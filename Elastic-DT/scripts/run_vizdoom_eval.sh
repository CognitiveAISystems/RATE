#!/bin/bash

# Evaluation script for Elastic-DT on ViZDoom Two Colors

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path> [num_episodes] [seed]"
    echo "Example: $0 checkpoints/vizdoom/edt_vizdoom_final_seed_42.pt 10 42"
    exit 1
fi

CHECKPOINT_PATH="$1"
NUM_EPISODES="${2:-10}"
SEED="${3:-42}"

echo "Evaluating Elastic-DT on ViZDoom Two Colors"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Number of episodes: $NUM_EPISODES"
echo "Seed: $SEED"

# * Initial parameters
# python Elastic-DT/scripts/eval_edt_vizdoom.py \
#     --checkpoint_path "$CHECKPOINT_PATH" \
#     --context_len 50 \
#     --n_blocks 4 \
#     --embed_dim 128 \
#     --n_heads 4 \
#     --dropout_p 0.1 \
#     --act_dim 5 \
#     --num_bin 60 \
#     --rtg_scale 1000 \
#     --target_return 56.5 \
#     --num_eval_episodes "$NUM_EPISODES" \
#     --episode_timeout 150 \
#     --seed "$SEED" \
#     --device cuda \
#     --save_results

# * Small parameters to run test
python Elastic-DT/scripts/eval_edt_vizdoom.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --context_len 50 \
    --n_blocks 2 \
    --embed_dim 32 \
    --n_heads 2 \
    --dropout_p 0.1 \
    --act_dim 5 \
    --num_bin 60 \
    --rtg_scale 1000 \
    --target_return 56.5 \
    --num_eval_episodes "$NUM_EPISODES" \
    --episode_timeout 150 \
    --seed "$SEED" \
    --device cuda \
    --save_results

echo "Evaluation complete!"

