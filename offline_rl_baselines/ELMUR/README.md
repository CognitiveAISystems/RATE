<h1 align="center">ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL</h1>
<h3 align="center">Offline RL agent for solving memory-intensive tasks with layer-local memory and Least Recently Used update/rewrite policy</h3>

<div align="center">
    <a href="https://openreview.net/forum?id=H2dvLYqlaa">
        <img src="https://img.shields.io/badge/arXiv-2306.09459-b31b1b.svg"/>
    </a>
    <a href="https://elmur-paper.github.io/">
        <img src="https://img.shields.io/badge/Website-Project_Page-blue.svg"/>
    </a>
    <a href="https://github.com/CognitiveAISystems/RATE">
        <img src="https://img.shields.io/badge/GitHub-RATE-green.svg"/>
    </a>
</div>

## Overview

**ELMUR (ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL)** is a novel memory-augmented transformer architecture specifically designed for handling long-term dependencies in sequential decision-making tasks. Unlike traditional transformers that struggle with very long sequences, ELMUR maintains layer-local memory embeddings that persists across time windows and Least Recently Used (LRU) memory update policy, allowing the model to effectively remember and reason over extended contexts.

> **Note**: ELMUR is integrated into the RATE training framework. For installation instructions and detailed task configurations, please see the main directory.


## Architecture

ELMUR processes tokens through two parallel tracks:

### Token Track
1. **Self-Attention**: Transformer-XL-style self-attention on current window tokens
2. **Memory Reading**: Cross-attention where tokens query the external memory (`mem2tok`)
3. **Feed-Forward**: Position-wise feed-forward network

### Memory Track
1. **Memory Writing**: Memory embeddings attend to current tokens to learn new information
2. **Memory Update**: Candidate memory updates are processed through feed-forward (`tok2mem`)
3. **LRU Replacement**: Memory embeddings are updated using Least Recently Used (LRU) policy with configurable blending

## Key Features

- **External Memory**: Fixed-size memory bank that persists across sequence windows
- **LRU Updates**: Efficient memory management using least-recently-used replacement
- **Relative Position Bias**: Optional relative positional encoding for cross-attention
- **Mixture of Experts (MoE)**: Support for multiple expert networks with load balancing
- **Multiple Position Encodings**: Compatible with RoPE, YaRN, ALiBi, relative, and absolute encodings

## Implementation Details

### Core Components

```
ELMUR/
├── model.py              # Main ELMURModel class
└── layers/
    ├── memory.py         # Memory state management
    ├── multi_head_attention.py  # Various attention mechanisms
    ├── positional_embeddings.py # Position encoding implementations
    ├── relative_bias.py  # Relative position bias for cross-attention
    ├── normalization.py  # Layer normalization variants
    ├── feed_forward_network.py  # Standard FFN implementation
    └── mixture_of_experts.py    # MoE implementation with DeepSeek-style shared experts
```

### Memory State Management

The memory system uses a `MemoryState` class that tracks:
- **Memory vectors**: `[B, M, D]` where B=batch size, M=memory size, D=model dimension
- **Memory positions**: `[B, M]` timestamps indicating when each slot was last updated

### Supported Position Encodings

- **Relative**: Transformer-XL style relative position embeddings (used by default)
- **RoPE**: Rotary Position Embedding for improved length extrapolation
- **YaRN**: Enhanced RoPE for very long sequences
- **ALiBi**: Attention with Linear Biases
- **Sinusoidal**: Classic sinusoidal position embeddings
- **Learnable**: Trainable absolute position embeddings

## Usage Examples

### Basic Usage

```python
from offline_rl_baselines.ELMUR.model import ELMURModel

# Initialize model
model = ELMURModel(
    state_dim=6,
    act_dim=8,
    d_model=32,
    n_layer=2,
    n_head=16,
    memory_size=32,
    pos_type="relative",
    use_moe=True,
    num_experts=1
)

# Initialize memory
memory_states = model.init_memory(batch_size=64, device=torch.device('cuda'))

# Forward pass
output = model(states, actions, rtgs, target, timesteps, memory_states=memory_states)
```

### Experiments

**T-Maze**
```bash
# context size = 10, 3 segments during training
python src/train.py --data.gamma=1 --data.max-length=None --data.path-to-dataset=None --dtype=float32 --end-seed=5 --max-n-final=3 --min-n-final=1 --model-mode=ELMUR --model.act-dim=4 --model.d-ff=128 --model.d-model=128 --model.detach-memory=True --model.dropatt=0.17 --model.dropout=0.10 --model.env-name=tmaze --model.label-smoothing=0.16 --model.load-balancing-loss-coef=0.1 --model.lru-blend-alpha=0.79 --model.max-seq-len=1024 --model.memory-dropout=0.01 --model.memory-init-std=0.001 --model.memory-size=2 --model.n-head=2 --model.n-layer=2 --model.n-shared-experts=2 --model.num-experts=2 --model.padding-idx=-10 --model.pos-type=relative --model.pre-lnorm=False --model.routed-d-ff=32 --model.sequence-format=s --model.shared-d-ff=512 --model.state-dim=4 --model.top-k=3 --model.use-causal-self-attn-mask=True --model.use-lru=True --model.use-moe=True --model.use-shared-expert=True --model.use-swiglu=False --online-inference.best_checkpoint_metric=Success_rate_9600 --start-seed=1 --tensorboard-dir=runs/TMaze/ELMUR/T_30 --text=my-experiment --training.batch-size=128 --training.beta-1=0.95 --training.beta-2=0.999 --training.ckpt-epoch=200 --training.context-length=10 --training.epochs=1000 --training.final-tokens=10000000 --training.grad-norm-clip=5 --training.learning-rate=0.00021 --training.log-last-segment-loss-only=True --training.lr-end-factor=1 --training.online-inference=True --training.sections=3 --training.use-cosine-decay=True --training.warmup-steps=10000 --training.weight-decay=0.0001 --wandb.project-name=ELMUR-T-Maze --wandb.wwandb=True & 
```

**MIKASA-Robo**
```bash
# RememberColor3-v0
python3 src/train.py --data.gamma=1 --data.path-to-dataset=datasets/data_mikasa_robo/MIKASA-Robo/unbatched/RememberColor3-v0 --dtype=float32 --end-seed=3 --model-mode=ELMUR --model.act-dim=8 --model.d-ff=128 --model.d-model=128 --model.detach-memory=True --model.dropatt=0.30 --model.dropout=0.13 --model.env-name=mikasa_robo_RememberColor3-v0 --model.label-smoothing=0.21 --model.load-balancing-loss-coef=0.1 --model.lru-blend-alpha=0.41 --model.max-seq-len=1024 --model.memory-dropout=0.05 --model.memory-init-std=0.1 --model.memory-size=256 --model.n-head=16 --model.n-layer=4 --model.n-shared-experts=1 --model.num-experts=16 --model.padding-idx=None --model.pos-type=relative --model.pre-lnorm=False --model.routed-d-ff=128 --model.sequence-format=s --model.shared-d-ff=128 --model.state-dim=6 --model.top-k=2 --model.use-causal-self-attn-mask=True --model.use-lru=True --model.use-moe=True --model.use-shared-expert=True --model.use-swiglu=False --online-inference.best_checkpoint_metric=success_once --online-inference.desired-return-1=60 --online-inference.episode-timeout=60 --online-inference.use-argmax=True --start-seed=1 --tensorboard-dir=runs/MIKASA_Robo/RememberColor3-v0 --text=my-experiment --training.batch-size=64 --training.beta-1=0.99 --training.beta-2=0.99 --training.ckpt-epoch=20 --training.context-length=20 --training.epochs=200 --training.final-tokens=10000000 --training.grad-norm-clip=5 --training.learning-rate=0.0002 --training.log-last-segment-loss-only=False --training.lr-end-factor=0.1 --training.online-inference=True --training.sections=3 --training.use-cosine-decay=True --training.warmup-steps=30000 --training.weight-decay=0.001 --wandb.project-name=ELMUR-MIKASA-Robo --wandb.wwandb=True

# RememberColor5-v0
python3 src/train.py --data.gamma=1 --data.path-to-dataset=datasets/data_mikasa_robo/MIKASA-Robo/unbatched/RememberColor5-v0 --dtype=float32 --end-seed=3 --model-mode=ELMUR --model.act-dim=8 --model.d-ff=128 --model.d-model=128 --model.detach-memory=True --model.dropatt=0.30 --model.dropout=0.13 --model.env-name=mikasa_robo_RememberColor5-v0 --model.label-smoothing=0.21 --model.load-balancing-loss-coef=0.1 --model.lru-blend-alpha=0.41 --model.max-seq-len=1024 --model.memory-dropout=0.05 --model.memory-init-std=0.1 --model.memory-size=256 --model.n-head=16 --model.n-layer=4 --model.n-shared-experts=1 --model.num-experts=16 --model.padding-idx=None --model.pos-type=relative --model.pre-lnorm=False --model.routed-d-ff=128 --model.sequence-format=s --model.shared-d-ff=128 --model.state-dim=6 --model.top-k=2 --model.use-causal-self-attn-mask=True --model.use-lru=True --model.use-moe=True --model.use-shared-expert=True --model.use-swiglu=False --online-inference.best_checkpoint_metric=success_once --online-inference.desired-return-1=60 --online-inference.episode-timeout=60 --online-inference.use-argmax=True --start-seed=1 --tensorboard-dir=runs/MIKASA_Robo/RememberColor5-v0 --text=my-experiment --training.batch-size=64 --training.beta-1=0.99 --training.beta-2=0.99 --training.ckpt-epoch=20 --training.context-length=20 --training.epochs=200 --training.final-tokens=10000000 --training.grad-norm-clip=5 --training.learning-rate=0.0002 --training.log-last-segment-loss-only=False --training.lr-end-factor=0.1 --training.online-inference=True --training.sections=3 --training.use-cosine-decay=True --training.warmup-steps=30000 --training.weight-decay=0.001 --wandb.project-name=ELMUR-MIKASA-Robo --wandb.wwandb=True

# RememberColor9-v0
python3 src/train.py --data.gamma=1 --data.path-to-dataset=datasets/data_mikasa_robo/MIKASA-Robo/unbatched/RememberColor9-v0 --dtype=float32 --end-seed=3 --model-mode=ELMUR --model.act-dim=8 --model.d-ff=128 --model.d-model=128 --model.detach-memory=True --model.dropatt=0.30 --model.dropout=0.13 --model.env-name=mikasa_robo_RememberColor9-v0 --model.label-smoothing=0.21 --model.load-balancing-loss-coef=0.1 --model.lru-blend-alpha=0.41 --model.max-seq-len=1024 --model.memory-dropout=0.05 --model.memory-init-std=0.1 --model.memory-size=256 --model.n-head=16 --model.n-layer=4 --model.n-shared-experts=1 --model.num-experts=16 --model.padding-idx=None --model.pos-type=relative --model.pre-lnorm=False --model.routed-d-ff=128 --model.sequence-format=s --model.shared-d-ff=128 --model.state-dim=6 --model.top-k=2 --model.use-causal-self-attn-mask=True --model.use-lru=True --model.use-moe=True --model.use-shared-expert=True --model.use-swiglu=False --online-inference.best_checkpoint_metric=success_once --online-inference.desired-return-1=60 --online-inference.episode-timeout=60 --online-inference.use-argmax=True --start-seed=1 --tensorboard-dir=runs/MIKASA_Robo/RememberColor9-v0 --text=my-experiment --training.batch-size=64 --training.beta-1=0.99 --training.beta-2=0.99 --training.ckpt-epoch=20 --training.context-length=20 --training.epochs=200 --training.final-tokens=10000000 --training.grad-norm-clip=5 --training.learning-rate=0.0002 --training.log-last-segment-loss-only=False --training.lr-end-factor=0.1 --training.online-inference=True --training.sections=3 --training.use-cosine-decay=True --training.warmup-steps=30000 --training.weight-decay=0.001 --wandb.project-name=ELMUR-MIKASA-Robo --wandb.wwandb=True

# TakeItBack-v0
python3 src/train.py --data.gamma=1 --data.path-to-dataset=datasets/data_mikasa_robo/MIKASA-Robo/unbatched/TakeItBack-v0 --dtype=float32 --end-seed=3 --model-mode=ELMUR --model.act-dim=8 --model.d-ff=128 --model.d-model=32 --model.detach-memory=True --model.dropatt=0.18 --model.dropout=0.014 --model.env-name=mikasa_robo_TakeItBack-v0 --model.label-smoothing=0.19 --model.load-balancing-loss-coef=0.05 --model.lru-blend-alpha=0.19 --model.max-seq-len=1024 --model.memory-dropout=0.23 --model.memory-init-std=0.001 --model.memory-size=32 --model.n-head=16 --model.n-layer=2 --model.n-shared-experts=1 --model.num-experts=1 --model.padding-idx=None --model.pos-type=relative --model.pre-lnorm=False --model.routed-d-ff=256 --model.sequence-format=s --model.shared-d-ff=32 --model.state-dim=6 --model.top-k=1 --model.use-causal-self-attn-mask=True --model.use-lru=True --model.use-moe=True --model.use-shared-expert=True --model.use-swiglu=False --online-inference.best_checkpoint_metric=success_once --online-inference.desired-return-1=180 --online-inference.episode-timeout=180 --online-inference.use-argmax=True --start-seed=1 --tensorboard-dir=runs/MIKASA_Robo/TakeItBack-v0 --text=my-experiment --training.batch-size=64 --training.beta-1=0.9 --training.beta-2=0.99 --training.ckpt-epoch=20 --training.context-length=60 --training.epochs=300 --training.final-tokens=20000000 --training.grad-norm-clip=1 --training.learning-rate=0.00026 --training.log-last-segment-loss-only=False --training.lr-end-factor=0.01 --training.online-inference=True --training.sections=3 --training.use-cosine-decay=False --training.warmup-steps=30000 --training.weight-decay=0.01 --wandb.project-name=ELMUR-MIKASA-Robo --wandb.wwandb=True
```

**POPGym-48**
```bash
# Training script for POPGym-48 tasks. Shown here is AutoencodeEasy-v0 config. For other tasks, refer to RATE training configs.
python src/train.py --data.gamma=1 --data.max-length=105 --data.path-to-dataset=data/POPGym/popgym-AutoencodeEasy-v0 --dtype=float32 --end-seed=3 --model-mode=ELMUR --model.act-dim=4 --model.d-ff=128 --model.d-model=64 --model.detach-memory=True --model.dropatt=0.26 --model.dropout=0.14 --model.env-name=popgym-AutoencodeEasy --model.label-smoothing=0.22 --model.load-balancing-loss-coef=0.1 --model.lru-blend-alpha=0.79 --model.max-seq-len=1024 --model.memory-dropout=0.17 --model.memory-init-std=0 --model.memory-size=8 --model.n-head=4 --model.n-layer=12 --model.n-shared-experts=2 --model.norm-type=rmsnorm --model.num-experts=1 --model.padding-idx=-10 --model.pos-type=relative --model.pre-lnorm=False --model.routed-d-ff=128 --model.sequence-format=s --model.shared-d-ff=256 --model.state-dim=-1 --model.top-k=1 --model.use-causal-self-attn-mask=True --model.use-lru=True --model.use-moe=True --model.use-shared-expert=True --model.use-swiglu=False --online-inference.best_checkpoint_metric=ReturnsMean_1.0 --online-inference.desired-return-1=1 --online-inference.episode-timeout=1001 --online-inference.use-argmax=False --start-seed=1 --tensorboard-dir=runs/POPGym/AutoencodeEasy-v0 --text=iclr-2026 --training.batch-size=128 --training.beta-1=0.99 --training.beta-2=0.99 --training.ckpt-epoch=50 --training.context-length=35 --training.epochs=800 --training.final-tokens=10000000 --training.grad-norm-clip=5 --training.learning-rate=0.00012 --training.log-last-segment-loss-only=False --training.lr-end-factor=0.01 --training.online-inference=True --training.sections=3 --training.use-cosine-decay=False --training.warmup-steps=50000 --training.weight-decay=0.1 --wandb.project-name=ELMUR-POPGym --wandb.wwandb=True
```

## Configuration Options

### Memory Configuration
- `memory_size`: Number of memory embeddings (default: 16)
- `memory_init_std`: Standard deviation for memory initialization (default: 0.02)
- `use_lru`: Use LRU replacement policy (default: True)
- `lru_blend_alpha`: Blending factor for LRU updates (default: 0.05)

### Position Encoding
- `pos_type`: Position encoding type (`relative`, `rope`, `yarn`, `alibi`, `sinusoidal`, `learnable`)
- `max_seq_len`: Maximum sequence length for positional embeddings

### Mixture of Experts
- `use_moe`: Enable Mixture of Experts (default: True)
- `num_experts`: Number of expert networks (default: 8)
- `top_k`: Number of experts to select per token (default: 2)
- `use_shared_expert`: Include always-active shared expert (default: True)
- `n_shared_experts`: Number of shared experts (default: 1)
- `shared_d_ff`: Shared expert dim
- `routed_d_ff`: Routed expert dim



## Citation

If you use ELMUR in your research, please cite:

```bibtex
@inproceedings{cherepanov2025elmur,
    title={{ELMUR}: External Layer Memory with Update/Rewrite for Long-Horizon {RL}},
    author={Egor Cherepanov and Alexey Kovalev and Aleksandr Panov},
    booktitle={CoRL 2025 Workshop RemembeRL},
    year={2025},
    url={https://openreview.net/forum?id=H2dvLYqlaa}
}
```

```bibtex
@misc{cherepanov2024recurrentactiontransformermemory,
      title={Recurrent Action Transformer with Memory},
      author={Egor Cherepanov and Alexey Staroverov and Dmitry Yudin and Alexey K. Kovalev and Aleksandr I. Panov},
      year={2024},
      eprint={2306.09459},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2306.09459},
}
```

## License

This implementation is part of the RATE framework and follows the same MIT License.
