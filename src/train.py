from dataclasses import dataclass, field, asdict
from typing import Optional
import tyro
from tyro.conf import FlagConversionOff
import wandb
import os, sys
from coolname import generate_slug

from src.utils.reconfigure_config import configure_model_architecture
from src.utils.set_seed import set_seed
from src.utils.dataloaders import create_dataloader
from src.utils.get_intro import IntroRenderer


from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.trainer import Trainer

import yaml
with open("wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']


@dataclass
class WandbConfig:
    project_name: str = "v2-RATE-ViZDoom2C"
    wwandb: FlagConversionOff[bool] = True

@dataclass
class DataConfig:
    gamma: float = 1.0
    path_to_dataset: Optional[str] = None
    max_length: Optional[int] = None

@dataclass
class TrainingConfig:
    # --training.sections=3
    learning_rate: float = 3e-4
    lr_end_factor: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.95
    weight_decay: float = 0.1
    batch_size: int = 64
    warmup_steps: int = 10000
    final_tokens: int = 10000000
    grad_norm_clip: float = 1.0
    epochs: int = 100
    ckpt_epoch: int = 8
    online_inference: FlagConversionOff[bool] = True
    log_last_segment_loss_only: FlagConversionOff[bool] = False
    use_cosine_decay: FlagConversionOff[bool] = True
    context_length: int = 30  # if RATE/GRATE: L = L, if DT: L = sections * L
    sections: int = 3        # if RATE/GRATE: S = S, if DT: S = 1

@dataclass
class ModelConfig:
    env_name: str = "vizdoom"
    state_dim: int = 3
    act_dim: int = 5
    n_layer: int = 6
    n_head: int = 8
    n_head_ca: int = 2
    d_model: int = 128
    d_head: int = 128
    d_inner: int = 128
    dropout: float = 0.2
    dropatt: float = 0.05
    mem_len: int = 300
    ext_len: int = 0
    num_mem_tokens: int = 5
    mem_at_end: FlagConversionOff[bool] = True
    mrv_act: str = 'relu'
    skip_dec_ffn: FlagConversionOff[bool] = False # toggled -> True else False
    padding_idx: Optional[int] = None

@dataclass
class OnlineInferenceConfig:
    use_argmax: FlagConversionOff[Optional[bool]] = None
    episode_timeout: Optional[int] = None
    desired_return_1: Optional[float] = None

@dataclass
class Config:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    online_inference: OnlineInferenceConfig = field(default_factory=OnlineInferenceConfig)
    tensorboard_dir: str = "runs/ViZDoom"

    model_mode: str = "RATE"
    arch_mode: str = "TrXL"
    start_seed: int = 1
    end_seed: int = 6
    text: str = ""

    # for t-maze only!
    min_n_final: Optional[int] = None
    max_n_final: Optional[int] = None

def add_env_specific_info_to_config(config):
    if config["model"]["env_name"] == "tmaze":
        config["training"]["max_segments"] = config["max_n_final"]
        config["online_inference"]["episode_timeout"] = \
            config["max_n_final"] * config["training"]["context_length"]
        config["online_inference"]["corridor_length"] = \
            config["max_n_final"] * config["training"]["context_length"] - 2
        config["training"]["sections"] = config["max_n_final"]
        if config["model_mode"] not in ["DT", "DTXL"]:
            config["training"]["sections"] = config["max_n_final"]
        else:
            config["training"]["sections"] = 1
    elif config["model"]["env_name"] == "memory_maze":
        config["data"]["only_non_zero_rewards"] = True
    elif "popgym" in config["model"]["env_name"]:
        config["training"]["max_segments"] = config["training"]["sections"]

    return config


if __name__ == "__main__":
    config = tyro.cli(Config)
    config = asdict(config)

    renderer = IntroRenderer()
    renderer.render_intro(config["model"]["env_name"])

    config["arctitecture_mode"] = config['arch_mode']
    config['text_description'] = config['text']

    config["experiment_codename"] = generate_slug(2).replace("-", "_")
    print("\033[1;92mExperiment unique codename: {}\033[0m".format(config['experiment_codename']))

    for RUN in range(config['start_seed'], config['end_seed']+1):
        set_seed(RUN)
        print(f"Random seed set as {RUN}")

        SEGMENT_LENGTH = config["training"]["context_length"]

        max_segments, max_length = configure_model_architecture(config)
        config = add_env_specific_info_to_config(config)

        config["group_name"] = f"exp_{config['text']}_model_{config['model_mode']}_arch_{config['arch_mode']}"
        config['run_name'] = f"{config['group_name']}_RUN_{RUN}"

        if config["data"]["max_length"] is None:
            max_length = max_length
        else:
            max_length = config["data"]["max_length"]

        print(f"Max length: {max_length}")

        train_dataloader = create_dataloader(config, max_length, SEGMENT_LENGTH)

        trainer = Trainer(config)
        model = trainer.train(train_dataloader)
        trainer.cleanup()