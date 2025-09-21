#!/bin/bash

nvidia-smi || echo "nvidia-smi not available"
# pip install -e .
# pip install wandb==0.18.5

# Run the main script for CartPole-v1
# CartPole-v1 configuration:
# - State space: Box([-4.8 -inf -0.41887903 -inf], [4.8 inf 0.41887903 inf], (4,), float32)
# - Action space: Discrete(2)
# - Episode length: typically ~200 steps, max 500
# - Reward: +1 for each step the pole remains upright

python src/train.py \
--model.norm-type=layer \
--data.gamma=0.99 \
--data.max-length=None \
--data.path-to-dataset=data/MDP/CartPole-v1 \
--dtype=float32 \
--end-seed=3 \
--start-seed=1 \
--model-mode=MATL \
--model.act-dim=2 \
--model.d-ff=256 \
--model.d-model=128 \
--model.detach-memory=True \
--model.dropatt=0.1 \
--model.dropout=0.1 \
--model.env-name=CartPole-v1 \
--model.label-smoothing=0.0 \
--model.load-balancing-loss-coef=0.01 \
--model.lru-blend-alpha=0.9 \
--model.max-seq-len=1024 \
--model.memory-dropout=0.05 \
--model.memory-init-std=0.01 \
--model.memory-size=16 \
--model.n-head=4 \
--model.n-layer=4 \
--model.n-shared-experts=1 \
--model.num-experts=4 \
--model.padding-idx=-10 \
--model.pos-type=relative \
--model.pre-lnorm=True \
--model.routed-d-ff=128 \
--model.sequence-format=s \
--model.shared-d-ff=256 \
--model.state-dim=4 \
--model.top-k=2 \
--model.use-causal-self-attn-mask=True \
--model.use-lru=True \
--model.use-moe=False \
--model.use-shared-expert=False \
--model.use-swiglu=True \
--model.use-shared-memory=False \
--model.use-relative-bias=True \
--model.use-tok2mem=True \
--model.use-mem2tok=True \
--online-inference.best_checkpoint_metric=mean_return \
--online-inference.episode-timeout=500 \
--online-inference.desired-return-1=500.0 \
--online-inference.use-argmax=True \
--tensorboard-dir=runs/MDP/CartPole-v1/MATL \
--text=cartpole-mdp-baseline \
--training.batch-size=512 \
--training.beta-1=0.9 \
--training.beta-2=0.999 \
--training.ckpt-epoch=2 \
--training.context-length=30 \
--training.epochs=100 \
--training.final-tokens=1000000 \
--training.grad-norm-clip=1.0 \
--training.learning-rate=3e-4 \
--training.log-last-segment-loss-only=False \
--training.lr-end-factor=0.1 \
--training.online-inference=True \
--training.sections=3 \
--training.use-cosine-decay=True \
--training.warmup-steps=1000 \
--training.weight-decay=0.01 \
--wandb.project-name=MATL-MDP-CartPole \
--wandb.wwandb=False
