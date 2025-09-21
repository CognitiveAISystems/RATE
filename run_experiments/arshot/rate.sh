#!/bin/bash

# ARShot Environment Training Script for RATE
# This script trains RATE model on the ARShot associative retrieval task

python src/train.py \
    --wandb.project-name "RATE-ARShot" \
    --wandb.wwandb False \
    \
    --model.env-name "arshot" \
    --n-pairs 6 \
    --shot-mode "after_pairs" \
    --deterministic-vocab True \
    --full-universe-vocab True \
    --randomize-pairs True \
    --include-pass-token True \
    --num-episodes 1000 \
    \
    --model.state-dim 3849 \
    --model.act-dim 3849 \
    --model.n-layer 6 \
    --model.n-head 8 \
    --model.n-head-ca 2 \
    --model.d-model 128 \
    --model.d-head 128 \
    --model.d-inner 128 \
    --model.dropout 0.2 \
    --model.dropatt 0.05 \
    --model.mem-len 300 \
    --model.ext-len 0 \
    --model.num-mem-tokens 5 \
    --model.mem-at-end True \
    --model.mrv-act "relu" \
    --model.skip-dec-ffn True \
    --model.padding-idx -10 \
    \
    --data.gamma 1.0 \
    \
    --training.learning-rate 3e-4 \
    --training.lr-end-factor 0.1 \
    --training.beta-1 0.9 \
    --training.beta-2 0.95 \
    --training.weight-decay 0.1 \
    --training.batch-size 128 \
    --training.warmup-steps 10000 \
    --training.final-tokens 10000000 \
    --training.grad-norm-clip 1.0 \
    --training.epochs 50 \
    --training.ckpt-epoch 5 \
    --training.online-inference True \
    --training.log-last-segment-loss-only False \
    --training.use-cosine-decay True \
    --training.context-length 12 \
    --training.sections 3 \
    \
    --online-inference.use-argmax True \
    --online-inference.episode-timeout 150 \
    --online-inference.desired-return-1 1.0 \
    --online-inference.best-checkpoint-metric "success_rate" \
    \
    --model-mode "RATE" \
    --arch-mode "TrXL" \
    --start-seed 1 \
    --end-seed 3 \
    --text "arshot_n6_after_pairs" \
    --dtype "float32"
