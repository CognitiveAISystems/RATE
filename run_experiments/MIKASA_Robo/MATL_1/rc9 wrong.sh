#!/bin/bash

# * MIKASA-Robo
python3 src/train.py \
    --model.norm-type=rmsnorm \
    --start-seed=1 \
    --end-seed=1 \
    --dtype=float32 \
    --data.gamma=1 \
    --data.path-to-dataset="/workspace-SR006.nfs2/echerepanov/datasets/data_mikasa_robo/MIKASA-Robo/unbatched/RememberColor9-v0" \
    --model-mode=MATL \
    --model.sequence-format=s \
    --model.state-dim=6 \
    --model.act-dim=8 \
    --model.d-model=64 \
    --model.d-ff=512 \
    --model.memory-size=16 \
    --model.max-seq-len=8192 \
    --model.memory-init-std=0.1 \
    --model.detach-memory=True \
    --model.use-causal-self-attn-mask=True \
    --model.use-lru=True \
    --model.lru-blend-alpha=0.99 \
    --model.pre-lnorm=False \
    --model.pos-type=relative \
    --model.train-stride=10 \
    --training.context-length=20 \
    --training.sections=3 \
    --model.dropout=0.2 \
    --model.dropatt=0.1 \
    --model.memory-dropout=0.3 \
    --model.label-smoothing=0.2 \
    --model.env-name='mikasa_robo_RememberColor9-v0' \
    --model.n-head=2 \
    --model.n-layer=8 \
    --model.padding-idx=None \
    --online-inference.use-argmax=False \
    --online-inference.episode-timeout=90 \
    --online-inference.desired-return-1=90 \
    --online-inference.best_checkpoint_metric='success_once' \
    --tensorboard-dir='runs/MIKASA_Robo/RememberColor9-v0' \
    --text='iclr-2025' \
    --training.batch-size=96 \
    --training.beta-1=0.9 \
    --training.beta-2=0.999 \
    --training.ckpt-epoch=25 \
    --training.epochs=250 \
    --training.final-tokens=10_000_000 \
    --training.grad-norm-clip=3.0 \
    --training.learning-rate=0.001 \
    --training.log-last-segment-loss-only=False \
    --training.lr-end-factor=0.01 \
    --training.online-inference=True \
    --training.use-cosine-decay=False \
    --training.warmup-steps=10_000 \
    --training.weight-decay=0.001 \
    --wandb.project-name='MATL-MIKASA-Robo' \
    --wandb.wwandb=True\
    --model.use-moe=True \
    --model.num-experts=4 \
    --model.top-k=2 \
    --model.use-shared-expert=True \
    --model.n-shared-experts=1 \
    --model.shared-d-ff=256 \
    --model.routed-d-ff=64 \
    --model.use-swiglu=False \
    --model.load-balancing-loss-coef=0.1
    
    
    
