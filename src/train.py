from dataclasses import dataclass, field, asdict
from typing import Optional, List
import tyro
from tyro.conf import FlagConversionOff
import wandb
import os, sys
from coolname import generate_slug

from src.utils.reconfigure_config import configure_model_architecture, add_env_specific_info_to_config
from src.utils.set_seed import set_seed
from src.utils.dataloaders import create_dataloader
from src.utils.get_intro import IntroRenderer
from src.trainer import Trainer

import yaml
if os.path.exists("wandb_config.yaml"):
    print("wandb_config.yaml exists, loading API key...")
    with open("wandb_config.yaml") as f:
        wandb_config = yaml.load(f, Loader=yaml.FullLoader)
    os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']
else:
    print("wandb_config.yaml does not exist, using default user API key...")


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    # Name of the W&B project to log to
    project_name: str = "RATE-MIKASA-Robo"
    # Toggle W&B logging on/off
    wwandb: FlagConversionOff[bool] = True

@dataclass
class DataConfig:
    """Configuration for dataset and data processing."""
    # Discount factor
    gamma: float = 1.0
    # Path to the dataset directory. Do not required for T-Maze (autogeneration)
    path_to_dataset: Optional[str] = None
    # Maximum sequence length (number of timesteps with (R, o, a)) to use from dataset.
    max_length: Optional[int] = None

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Learning rate
    learning_rate: float = 3e-4
    # Factor to reduce learning rate by at the end of training
    lr_end_factor: float = 0.1
    # Beta 1 for Adam optimizer
    beta_1: float = 0.9
    # Beta 2 for Adam optimizer
    beta_2: float = 0.95
    # Weight decay for Adam optimizer
    weight_decay: float = 0.1
    # Batch size
    batch_size: int = 64
    # Number of processed tokens during warmup
    warmup_steps: int = 10000
    # Number of final tokens for cosine learning rate decay
    final_tokens: int = 10000000
    # Gradient norm clip
    grad_norm_clip: float = 1.0
    # Number of epochs
    epochs: int = 100
    # Checkpoint epoch
    ckpt_epoch: int = 8
    # Toggle online inference. If True, will perform online inference after each ckpt_epoch.
    online_inference: FlagConversionOff[bool] = True
    # Toggle logging last segment loss only. If true, will only log loss of last segment.
    log_last_segment_loss_only: FlagConversionOff[bool] = False
    # Toggle cosine learning rate decay. If true, will use cosine learning rate decay, else linear.
    use_cosine_decay: FlagConversionOff[bool] = True
    # Context length. If RATE: K = K, if DT: K = sections * K
    context_length: int = 30
    # Number of sections the sequence is split into.
    sections: int = 3
    """
    How to interpret the context length and sections?
        1. Let's say we have a sequence of length 90.
        2. If we set context_length = 30 and sections = 3, then the sequence will be split into 3 sections of 30 timesteps each.
        3. If model_mode in ["RATE", "RMT", "TrXL"], then the context length is the actual context length,
            and effectoive context length is context_length * sections. In this case, the model will sequentially
            process 3 sections of 30 timesteps each, therefore the effective context length is 90.
        4. If model_mode in ["DT", "DTXL","BC", "CQL", "IQL"], then the context length is context_length * sections,
            and the model will process the entire sequence at once.

    Conclusion:
        - If you want to compare the performance of, for instance, RATE and DT using the same trajectories of length 90,
        you should set: 
            - context_length = 30 and sections = 3 or context_length = 90 and sections = 1 for DT
            - context_length = 30 and sections = 3 for RATE
    """

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Name of the environment
    env_name: str = "vizdoom"
    # State dimension
    state_dim: int = 3
    # Action dimension / number of actions if dimension is 1
    act_dim: int = 5
    # Number of transformer layers (default: 6)
    n_layer: Optional[int] = None
    # Number of attention heads (default: 8)
    n_head: Optional[int] = None
    # Number of MRV cross-attention heads (default: 2)
    n_head_ca: Optional[int] = None
    # Model dimension (default: 128)
    d_model: int = 128
    # Dimension of each attention head (default: 128)
    d_head: Optional[int] = None
    # Dimension of the inner feed-forward network (default: 128)
    d_inner: Optional[int] = None
    # Dropout rate (default: 0.2)
    dropout: float = 0.2
    # Dropout rate for attention (default: 0.05)
    dropatt: Optional[float] = None
    # Memory length from TransformerXL (number of cached hidden states) (default: 300)
    mem_len: Optional[int] = None
    # Extended context length from TransformerXL (default: 0)
    ext_len: Optional[int] = None
    # Number of memory tokens (default: 5)
    num_mem_tokens: Optional[int] = None
    # If True, memory tokens are added to the end of the sequence (default: True)
    mem_at_end: FlagConversionOff[bool] = True
    # Activation function for MRV (default: 'relu')
    mrv_act: Optional[str] = None
    # Skip the feed-forward network in the decoder (default: False, but always True for RATE)
    skip_dec_ffn: FlagConversionOff[bool] = False # toggled -> True else False
    # Index for padding tokens if dataset trajectories has different lengths. Recommended: -10
    padding_idx: Optional[int] = None
    # CQL alpha parameter (for CQL.py only!, default: 1.0)
    cql_alpha: Optional[float] = None # 1.0 (for CQL only)
    
    # Parameters for BC
    backbone: Optional[str] = None  # Choose between 'mlp' and 'lstm'
    lstm_layers: Optional[int] = None   # Number of LSTM layers
    bidirectional: FlagConversionOff[Optional[bool]] = None  # Use bidirectional LSTM
    reset_hidden_state_batch: FlagConversionOff[Optional[bool]] = None  # Reset hidden state for each batch

    # Parameters for DMamba
    token_mixer: Optional[str] = None  # ['mamba'] Choose between 'mamba' and 'attn'
    window_size: Optional[int] = None  # [4] Window size for convolutional token mixer
    conv_proj: FlagConversionOff[Optional[bool]] = None  # [True] Use convolutional token mixer projection

    # Parameters for LSDT
    kernel_size: Optional[int] = None  # [5] Kernel size for convolutional layer
    convdim: Optional[int] = None  # [128] Output channel size for convolutional layer

@dataclass
class OnlineInferenceConfig:
    """Configuration for online inference each ckpt_epoch during training."""
    # Use argmax for action selection. If this option is unavailable, in the code the best option is used.
    use_argmax: FlagConversionOff[Optional[bool]] = None
    # Episode timeout in timesteps(default: None)
    episode_timeout: Optional[int] = None
    # Desired return (return-to-go) for final evaluation (default: None)
    desired_return_1: Optional[float] = None
    # Metric to use for best checkpoint selection (default: None)
    best_checkpoint_metric: Optional[str] = None

@dataclass
class Config:
    """Main configuration class combining all config components."""
    # W&B logging configuration
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # Dataset configuration
    data: DataConfig = field(default_factory=DataConfig)
    # Training parameters
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Model architecture configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    # Online inference configuration
    online_inference: OnlineInferenceConfig = field(default_factory=OnlineInferenceConfig)
    # Tensorboard directory
    tensorboard_dir: str = "runs/ViZDoom"
    # Model mode
    model_mode: str = "RATE"
    # Base architecture mode (TrXL, GTrXL, TrXL-I)
    arch_mode: str = "TrXL"
    # Starting seed (seed of the first model run)
    start_seed: int = 1
    # Ending seed (seed of the last model run)
    end_seed: int = 6
    # Text description (additional user-defined information of the experiment)
    text: str = ""

    # For T-Maze only! (specified in the .sh running scripts)
    min_n_final: Optional[int] = None
    max_n_final: Optional[int] = None


if __name__ == "__main__":
    """
    Main training script for RATE (Recurrent Action Transformer with Memory).
    (This code is also applicable to other models: DT, DT-XL, RMT, TrXL, BC, CQL, IQL)
    
    The script performs the following steps:
    1. Loads and processes configuration from command line arguments
    2. Initializes experiment tracking and logging
    3. Runs multiple training iterations with different random seeds
    4. For each seed:
        - Configures model architecture
        - Creates data loader
        - Trains the model
        - Performs cleanup
    """

    # Parse command line arguments using tyro and convert to dictionary
    config = tyro.cli(Config)
    config = asdict(config)

    # Initialize and display environment-specific introduction
    renderer = IntroRenderer()
    renderer.render_intro(config["model"]["env_name"])

    config["arctitecture_mode"] = config['arch_mode']
    config['text_description'] = config['text']

    # Generate a unique codename for the experiment
    config["experiment_codename"] = generate_slug(2).replace("-", "_")
    print("\033[1;92mExperiment unique codename: {}\033[0m".format(config['experiment_codename']))

    # Flag to ensure segment length is set only once
    switched = False

    # Run training for multiple seeds
    for RUN in range(config['start_seed'], config['end_seed']+1):
        # Set random seed for reproducibility
        set_seed(RUN)
        print(f"Random seed set as {RUN}")

        # Set segment length on first iteration
        if not switched:
            SEGMENT_LENGTH = config["training"]["context_length"]
            switched = True

        # Configure model architecture and get maximum sequence lengths
        max_segments, max_length = configure_model_architecture(config)
        
        # Add environment-specific configuration parameters
        config = add_env_specific_info_to_config(config)

        # Set up experiment naming for logging
        config["group_name"] = f"exp_{config['text']}_model_{config['model_mode']}_arch_{config['arch_mode']}"
        config['run_name'] = f"{config['group_name']}_RUN_{RUN}"

        if config["data"]["max_length"] is None:
            max_length = max_length
        else:
            max_length = config["data"]["max_length"]

        print(f"Max length: {max_length}")

        # Initialize data loader with configured parameters
        train_dataloader = create_dataloader(config, max_length, SEGMENT_LENGTH)
        
        # Initialize trainer and train the model
        trainer = Trainer(config)
        model = trainer.train(train_dataloader)

        # Perform cleanup after training
        trainer.cleanup()