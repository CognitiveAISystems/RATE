#!/bin/bash

# ARShot Environment Training Script for BC-LSTM
# This script trains BC-LSTM model on the ARShot associative retrieval task

python3 src/train.py \
    --wandb.project-name "MATL-ARShot" \
    --wandb.wwandb False \
    \
    --model.env-name "arshot" \
    --n-pairs 11 \
    --shot-mode "after_any_colon" \
    --deterministic-vocab True \
    --full-universe-vocab True \
    --randomize-pairs True \
    --include-pass-token True \
    --num-episodes 1000 \
    --max-vocab-size 500 \
    \
    --data.gamma 1.0 \
    --data.path-to-dataset None \
    \
    --training.learning-rate 0.0003 \
    --training.lr-end-factor 0.1 \
    --training.beta-1 0.9 \
    --training.beta-2 0.999 \
    --training.weight-decay 0.01 \
    --training.batch-size 256 \
    --training.warmup-steps 100 \
    --training.final-tokens 10000000 \
    --training.grad-norm-clip 5.0 \
    --training.epochs 100 \
    --training.ckpt-epoch 10 \
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
    --model.d-model 64 \
    --model.dropout 0.2 \
    --model.padding-idx -10 \
    --model.backbone lstm \
    --model.lstm_layers 1 \
    --model.bidirectional False \
    --model.reset_hidden_state_batch True \
    \
    --tensorboard-dir runs/ARShot/BC_LSTM \
    --model-mode BC \
    --start-seed 1 \
    --end-seed 3 \
    --dtype "float32" \
    --text arshot_n11_after_pairs
