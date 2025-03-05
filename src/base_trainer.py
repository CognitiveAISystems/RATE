import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from datetime import datetime


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wwandb = config["wandb"]["wwandb"]

        base_dir = config.get("tensorboard_dir", f"runs/{config['model']['env_name']}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.get("run_name", "run")
        group_name = config.get("group_name", "group")
        exp_codename = config.get("experiment_codename", "exp")
        self.run_dir = f"{base_dir}/{group_name}/{exp_codename}/{run_name}_{timestamp}"
        
        self.ckpt_dir = f"{self.run_dir}/checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.run_dir)
        
    def log(self, metrics, step=None):
        """Universal logging method for both WandB and TensorBoard"""
        if step is None:
            step = getattr(self, 'global_step', 0)
            if hasattr(self, 'global_step'):
                self.global_step += 1

        # Log to WandB
        if self.wwandb:
            wandb.log(metrics)
            
        # Log to TensorBoard
        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
                elif isinstance(value, torch.Tensor):
                    self.writer.add_scalar(key, value.item(), step)

    def cleanup(self):
        if hasattr(self, 'writer'):
            self.writer.close()