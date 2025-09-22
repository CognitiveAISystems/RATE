#!/bin/bash

# ARShot Environment Training Script for RATE
# This script trains RATE model on the ARShot associative retrieval task
    # --model.state-dim 3849 \
    # --model.act-dim 3849 \

python src/train.py \
    --wandb.project-name "MATL-ARShot" \
    --wandb.wwandb False \
    \
    --model.env-name "arshot" \
    --n-pairs 2 \
    --shot-mode "after_any_colon" \
    --deterministic-vocab True \
    --full-universe-vocab True \
    --randomize-pairs True \
    --include-pass-token True \
    --num-episodes 10 \
    --max-vocab-size 10 \
    \
    --model.state-dim 3849 \
    --model.act-dim 3849 \
    --model.n-layer 6 \
    --model.n-head 8 \
    --model.n-head-ca 0 \
    --model.d-model 64 \
    --model.d-head 64 \
    --model.d-inner 64 \
    --model.dropout 0.2 \
    --model.dropatt 0.05 \
    --model.mem-len 0 \
    --model.ext-len 0 \
    --model.num-mem-tokens 0 \
    --model.mem-at-end False \
    --model.mrv-act "no_act" \
    --model.skip-dec-ffn False \
    --model.padding-idx -10 \
    --tensorboard-dir=runs/ARShot/DT \
    \
    --data.gamma 1.0 \
    \
    --training.learning-rate 3e-4 \
    --training.lr-end-factor 0.1 \
    --training.beta-1 0.9 \
    --training.beta-2 0.95 \
    --training.weight-decay 0.1 \
    --training.batch-size 64 \
    --training.warmup-steps 10000 \
    --training.final-tokens 10000000 \
    --training.grad-norm-clip 1.0 \
    --training.epochs 1000 \
    --training.ckpt-epoch 100 \
    --training.online-inference True \
    --training.log-last-segment-loss-only False \
    --training.use-cosine-decay True \
    --training.context-length 20 \
    --training.sections 3 \
    \
    --online-inference.use-argmax True \
    --online-inference.episode-timeout 2505 \
    --online-inference.desired-return-1 1.0 \
    --online-inference.best-checkpoint-metric "success_rate" \
    \
    --model-mode "DT" \
    --arch-mode "TrXL" \
    --start-seed 1 \
    --end-seed 3 \
    --text "arshot_n11_after_pairs" \
    --dtype "float32"
