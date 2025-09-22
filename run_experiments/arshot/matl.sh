#!/bin/bash

# ARShot Environment Training Script for MATL
# This script trains MATL model on the ARShot associative retrieval task

python src/train.py \
    --wandb.project-name "MATL-ARShot" \
    --wandb.wwandb False \
    \
    --model.env-name "arshot" \
    --n-pairs 5 \
    --shot-mode "after_any_colon" \
    --deterministic-vocab True \
    --full-universe-vocab True \
    --randomize-pairs True \
    --include-pass-token True \
    --num-episodes 100 \
    --max-vocab-size 500 \
    \
    --model.state-dim 3849 \
    --model.act-dim 3849 \
    --model.n-layer 6 \
    --model.n-head 8 \
    --model.d-ff=256 \
    --model.d-model=128 \
    --model.dropout 0.2 \
    --model.dropatt 0.05 \
    --model.padding-idx -10 \
    \
    --model.norm-type=layer \
    --data.max-length=None \
    --model.detach-memory=True \
    --model.label-smoothing=0.0 \
    --model.load-balancing-loss-coef=0.01 \
    --model.lru-blend-alpha=0.05 \
    --model.max-seq-len=1024 \
    --model.memory-dropout=0.05 \
    --model.memory-init-std=0.01 \
    --model.memory-size=16 \
    --model.n-shared-experts=1 \
    --model.num-experts=4 \
    --model.pos-type=relative \
    --model.pre-lnorm=False \
    --model.routed-d-ff=128 \
    --model.sequence-format=sa \
    --model.shared-d-ff=256 \
    --model.top-k=2 \
    --model.use-causal-self-attn-mask=True \
    --model.use-lru=True \
    --model.use-moe=True \
    --model.use-shared-expert=True \
    --model.use-swiglu=False \
    --model.use-shared-memory=False \
    --model.use-relative-bias=True \
    --model.use-tok2mem=True \
    --model.use-mem2tok=True \
    --tensorboard-dir=runs/ARShot/MATL \
    \
    --data.gamma 1.0 \
    \
    --training.learning-rate 3e-4 \
    --training.lr-end-factor 0.1 \
    --training.beta-1 0.9 \
    --training.beta-2 0.999 \
    --training.weight-decay 0.1 \
    --training.batch-size 64 \
    --training.warmup-steps 10000 \
    --training.final-tokens 10000000 \
    --training.grad-norm-clip 3.0 \
    --training.epochs 1000 \
    --training.ckpt-epoch 100 \
    --training.online-inference True \
    --training.log-last-segment-loss-only False \
    --training.use-cosine-decay True \
    --training.context-length 30 \
    --training.sections 1 \
    \
    --online-inference.use-argmax False \
    --online-inference.episode-timeout 2505 \
    --online-inference.desired-return-1 1.0 \
    --online-inference.best-checkpoint-metric "success_rate" \
    \
    --model-mode "MATL" \
    --start-seed 1 \
    --end-seed 3 \
    --text "arshot_n11_after_pairs" \
    --dtype "bfloat16"