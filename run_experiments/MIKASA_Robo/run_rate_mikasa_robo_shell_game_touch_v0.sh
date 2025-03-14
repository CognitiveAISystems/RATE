#!/bin/bash

# * MIKASA-Robo
python3 src/train.py \
    --wandb.project-name='RATE-MIKASA-Robo' \
    --wandb.wwandb \
    --data.gamma=1.0 \
    --data.path-to-dataset='../../MIKASA-Robo/data/MIKASA-Robo/unbatched/ShellGameTouch-v0' \
    --training.learning-rate=0.0003 \
    --training.lr-end-factor=0.1 \
    --training.beta-1=0.9 \
    --training.beta-2=0.95 \
    --training.weight-decay=0.1 \
    --training.batch-size=64 \
    --training.warmup-steps=10_000 \
    --training.final-tokens=10_000_000 \
    --training.grad-norm-clip=1.0 \
    --training.epochs=100 \
    --training.ckpt-epoch=5 \
    --training.online-inference \
    --training.no-log-last-segment-loss-only \
    --training.use-cosine-decay \
    --training.context-length=30 \
    --training.sections=3 \
    --model.env-name='mikasa_robo_ShellGameTouch-v0' \
    --model.state-dim=6 \
    --model.act-dim=8 \
    --model.n-layer=4 \
    --model.n-head=6 \
    --model.n-head-ca=2 \
    --model.d-model=64 \
    --model.d-head=64 \
    --model.d-inner=64 \
    --model.dropout=0.2 \
    --model.dropatt=0.05 \
    --model.mem-len=0 \
    --model.ext-len=0 \
    --model.num-mem-tokens=5 \
    --model.mem-at-end \
    --model.mrv-act='relu' \
    --model.skip-dec-ffn \
    --model.padding-idx=None \
    --tensorboard-dir='runs/MIKASA_Robo/ShellGameTouch-v0' \
    --model-mode='RATE' \
    --arch-mode='TrXL' \
    --start-seed=1 \
    --end-seed=3 \
    --text='ret=1=constant' \
    --online-inference.use-argmax=False \
    --online-inference.episode-timeout=90 \
    --online-inference.desired-return-1=1

    # --model.n-layer=6 \
    # --model.n-head=8 \
    # --batch_size=64