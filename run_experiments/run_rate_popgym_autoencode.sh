#!/bin/bash

# * T-Maze
python3 src/train.py \
    --wandb.project-name='RATE-POPGym' \
    --wandb.wwandb=False \
    --data.gamma=1.0 \
    --data.path-to-dataset='data/POPGym/popgym-PositionOnlyCartPoleHard-v0/' \
    --data.max-length=66 \
    --training.learning-rate=0.0003 \
    --training.lr-end-factor=0.1 \
    --training.beta-1=0.9 \
    --training.beta-2=0.95 \
    --training.weight-decay=0.1 \
    --training.batch-size=64 \
    --training.warmup-steps=100 \
    --training.final-tokens=10_000_000 \
    --training.grad-norm-clip=1.0 \
    --training.epochs=50 \
    --training.ckpt-epoch=1 \
    --training.online-inference=True \
    --training.log-last-segment-loss-only=True \
    --training.use-cosine-decay=False \
    --training.context-length=3 \
    --training.sections=3 \
    --model.env-name='popgym-PositionOnlyCartPoleHard' \
    --model.state-dim=-1 \
    --model.act-dim=2 \
    --model.n-layer=8 \
    --model.n-head=10 \
    --model.n-head-ca=4 \
    --model.d-model=64 \
    --model.d-head=64 \
    --model.d-inner=64 \
    --model.dropout=0.05 \
    --model.dropatt=0.0 \
    --model.mem-len=0 \
    --model.ext-len=0 \
    --model.num-mem-tokens=5 \
    --model.mem-at-end=True \
    --model.mrv-act='relu' \
    --model.skip-dec-ffn=True \
    --model.padding-idx=-10 \
    --tensorboard-dir='runs/POPGym' \
    --model-mode='RATE' \
    --arch-mode='TrXL' \
    --start-seed=1 \
    --end-seed=10 \
    --text='' \
    --online-inference.use-argmax=False \
    --online-inference.episode-timeout=1001 \
    --online-inference.desired-return-1=0.0


# context_length = 64 // 3 + 1 = 22
# max_length = context_length * 3 = 66
#  --training.ckpt-epoch=10 