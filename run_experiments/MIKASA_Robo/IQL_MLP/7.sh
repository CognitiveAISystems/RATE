#!/bin/bash

# * MIKASA-Robo
python3 src/train.py \
    --wandb.project-name='RATE-MIKASA-Robo' \
    --wandb.wwandb=True \
    --data.gamma=1.0 \
    --data.path-to-dataset='data/MIKASA-Robo/unbatched/InterceptGrabMedium-v0' \
    --training.learning-rate=0.0001 \
    --training.lr-end-factor=0.1 \
    --training.beta-1=0.9 \
    --training.beta-2=0.999 \
    --training.weight-decay=0.1 \
    --training.batch-size=96 \
    --training.warmup-steps=10_000 \
    --training.final-tokens=10_000_000 \
    --training.grad-norm-clip=3.0 \
    --training.epochs=150 \
    --training.ckpt-epoch=25 \
    --training.online-inference=True \
    --training.log-last-segment-loss-only=False \
    --training.use-cosine-decay=False \
    --training.context-length=90 \
    --training.sections=1 \
    --model.env-name='mikasa_robo_InterceptGrabMedium-v0' \
    --model.state-dim=6 \
    --model.act-dim=8 \
    --model.d-model=128 \
    --model.dropout=0.0 \
    --model.padding-idx=None \
    --tensorboard-dir='runs/MIKASA_Robo/InterceptGrabMedium-v0' \
    --model-mode='IQL' \
    --arch-mode='MLP' \
    --start-seed=1 \
    --end-seed=3 \
    --text='IQL' \
    --online-inference.use-argmax=False \
    --online-inference.episode-timeout=90 \
    --online-inference.desired-return-1=90 \
    --online-inference.best_checkpoint_metric='success_once'