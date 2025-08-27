#!/bin/bash

# MATL Demo Script with MoE Support
# Supports: Standard FFN, SwiGLU FFN, and MoE with SwiGLU experts
# Sequence formats: s, sa, sra, sr

# MATL with MoE configuration
python3 src/train.py \
    --model.norm-type=rmsnorm \
    --start-seed=1 \
    --end-seed=1 \
    --dtype=bfloat16 \
    --data.gamma=1 \
    --data.max-length=None \
    --data.path-to-dataset=None \
    --model-mode=MATL \
    --model.sequence-format=s \
    --model.act-dim=4 \
    --model.d-model=64 \
    --model.d-ff=512 \
    --model.memory-size=4 \
    --model.max-seq-len=2048 \
    --model.memory-init-std=0.01 \
    --model.detach-memory=True \
    --model.use-causal-self-attn-mask=True \
    --model.use-lru=False \
    --model.lru-blend-alpha=0.5 \
    --model.pre-lnorm=False \
    --model.pos-type=learnable \
    --model.train-stride=10 \
    --training.context-length=10 \
    --training.sections=3 \
    --model.dropatt=0.0 \
    --model.dropout=0.1 \
    --model.memory-dropout=0.3 \
    --model.label-smoothing=0.2 \
    --model.env-name=tmaze \
    --min-n-final=1 \
    --max-n-final=3 \
    --model.n-head=8 \
    --model.n-layer=6 \
    --model.padding-idx=-10 \
    --model.state-dim=4 \
    --online-inference.best_checkpoint_metric=Success_rate_x50 \
    --tensorboard-dir=runs/TMaze/MATL/T_30 \
    --text=ffn+gelu \
    --training.batch-size=512 \
    --training.beta-1=0.95\
    --training.beta-2=0.99 \
    --training.ckpt-epoch=5 \
    --training.epochs=100 \
    --training.final-tokens=10000000 \
    --training.grad-norm-clip=1 \
    --training.learning-rate=0.0001 \
    --training.log-last-segment-loss-only=True \
    --training.lr-end-factor=0.01 \
    --training.online-inference=True \
    --training.use-cosine-decay=False \
    --training.warmup-steps=10000 \
    --training.weight-decay=0.01 \
    --wandb.project-name=MATL-T-Maze \
    --wandb.wwandb=True \
    --model.use-moe=True \
    --model.num-experts=8 \
    --model.top-k=2 \
    --model.use-swiglu=True \
    --model.load-balancing-loss-coef=0.01

# Alternative configurations:
# 
# Standard FFN with GELU:
# --model.use-moe=False --model.use-swiglu=False
#
# Standard FFN with SwiGLU:
# --model.use-moe=False --model.use-swiglu=True
#
# MoE with different expert counts:
# --model.use-moe=True --model.num-experts=16 --model.top-k=2