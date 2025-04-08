#!/bin/bash

# * MIKASA-Robo
python3 src/train.py \
    --wandb.project-name='RATE-MIKASA-Robo' \
    --wandb.wwandb=True \
    --data.gamma=1.0 \
    --data.path-to-dataset='data/MIKASA-Robo/unbatched/RememberColor9-v0' \
    --training.learning-rate=0.0003 \
    --training.lr-end-factor=0.1 \
    --training.beta-1=0.9 \
    --training.beta-2=0.999 \
    --training.weight-decay=0.1 \
    --training.batch-size=96 \
    --training.warmup-steps=10_000 \
    --training.final-tokens=10_000_000 \
    --training.grad-norm-clip=1.0 \
    --training.epochs=150 \
    --training.ckpt-epoch=25 \
    --training.online-inference=True \
    --training.log-last-segment-loss-only=False \
    --training.use-cosine-decay=True \
    --training.context-length=60 \
    --training.sections=1 \
    --model.env-name='mikasa_robo_RememberColor9-v0' \
    --model.state-dim=6 \
    --model.act-dim=8 \
    --model.d-model=64 \
    --model.dropout=0.0 \
    --model.padding-idx=None \
    --tensorboard-dir='runs/MIKASA_Robo/RememberColor9-v0' \
    --model-mode='BC' \
    --start-seed=1 \
    --end-seed=3 \
    --text='BC_MLP' \
    --online-inference.use-argmax=False \
    --online-inference.episode-timeout=60 \
    --online-inference.desired-return-1=60 \
    --online-inference.best_checkpoint_metric='success_once'