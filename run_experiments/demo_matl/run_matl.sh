#!/bin/bash


# s, sa, sra, sr

# RATE
python3 src/train.py \
    --model.norm-type=layer \
    --start-seed=1 \
    --end-seed=1 \
    --dtype=float32 \
    --data.gamma=1 \
    --data.max-length=None \
    --data.path-to-dataset=None \
    --model-mode=MATL \
    --model.sequence-format=sr \
    --model.act-dim=4 \
    --model.d-model=256 \
    --model.d-ff=1024 \
    --model.memory-size=128 \
    --model.max-seq-len=10_000 \
    --model.memory-init-std=0.01 \
    --model.detach-memory=True \
    --model.use-causal-self-attn-mask=True \
    --model.use-lru=True \
    --model.lru-blend-alpha=0.95 \
    --model.pre-lnorm=True \
    --model.pos-type=relative \
    --model.train-stride=10 \
    --training.context-length=10 \
    --training.sections=3 \
    --model.dropatt=0.1 \
    --model.dropout=0.2 \
    --model.memory-dropout=0.05 \
    --model.label-smoothing=0.00 \
    --model.env-name=tmaze \
    --min-n-final=1 \
    --max-n-final=3 \
    --model.n-head=4 \
    --model.n-layer=4 \
    --model.padding-idx=-10 \
    --model.state-dim=4 \
    --online-inference.best_checkpoint_metric=Success_rate_x50 \
    --tensorboard-dir=runs/TMaze/MATL/T_30 \
    --text=MATL \
    --training.batch-size=512 \
    --training.beta-1=0.9 \
    --training.beta-2=0.99 \
    --training.ckpt-epoch=5 \
    --training.epochs=200 \
    --training.final-tokens=10000000 \
    --training.grad-norm-clip=0.5 \
    --training.learning-rate=0.0003 \
    --training.log-last-segment-loss-only=True \
    --training.lr-end-factor=0.01 \
    --training.online-inference=True \
    --training.use-cosine-decay=True \
    --training.warmup-steps=20000 \
    --training.weight-decay=0.01 \
    --wandb.project-name=MATL-T-Maze \
    --wandb.wwandb=True