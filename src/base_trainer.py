import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from datetime import datetime
import json
import platform
import subprocess
import sys


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

        if self.wwandb:
            self.wandb_run = wandb.init(
                project=config['wandb']['project_name'],
                name=f"{run_name}_{timestamp}",
                group=group_name,
                config=config,
                save_code=True,
                reinit=True
            )
        
        self.ckpt_dir = f"{self.run_dir}/checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)

        run_metadata = {
            "timestamp": timestamp,
            "hostname": os.uname().nodename,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB" if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(),
            "git_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip(),
            "git_branch": subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip(),
            "git_dirty": bool(subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()),
            "command_line": " ".join(sys.argv),
            "working_directory": os.getcwd(),
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
            },
            "installed_packages": subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('ascii').split('\n'),
        }

        config["run_metadata"] = run_metadata

        # Save config to the run directory
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
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
        if self.wwandb and hasattr(self, 'wandb_run'):
            self.wandb_run.finish()