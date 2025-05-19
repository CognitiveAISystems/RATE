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
    """Base trainer class for model training with logging and checkpointing support.

    This class provides a foundation for training models with integrated support for:
    - TensorBoard logging
    - Weights & Biases (wandb) integration
    - Checkpoint management
    - Run metadata tracking
    - Best metric tracking
    - Environment and system information logging

    The trainer automatically creates a unique run directory for each training session
    and manages logging to both TensorBoard and wandb (if enabled).

    Args:
        config: Configuration dictionary containing training settings.
            Required keys:
            - model.env_name: Name of the environment (str)
            - wandb.wwandb: Whether to use wandb logging (bool)
            - wandb.project_name: Name of the wandb project (str)
            Optional keys:
            - tensorboard_dir: Base directory for TensorBoard logs (str)
            - run_name: Name of the current run (str)
            - group_name: Name of the experiment group (str)
            - experiment_codename: Code name for the experiment (str)
            - online_inference.best_checkpoint_metric: Metric name for best checkpoint (str)

    Attributes:
        config: Configuration dictionary.
        device: PyTorch device (cuda/cpu).
        wwandb: Whether wandb logging is enabled.
        run_dir: Directory for the current run.
        ckpt_dir: Directory for model checkpoints.
        best_metric_name: Name of the metric for best checkpoint tracking.
        best_metric_value: Best value of the tracked metric.
        best_checkpoint_path: Path to the best checkpoint.
        writer: TensorBoard SummaryWriter instance.
        wandb_run: wandb run instance (if wandb is enabled).

    Notes:
        - Creates a unique run directory with timestamp
        - Automatically tracks system and environment information
        - Supports both TensorBoard and wandb logging
        - Manages best checkpoint based on specified metric
        - Saves complete configuration and metadata
    """

    def __init__(self, config: dict):
        """Initialize the base trainer with configuration and setup logging.

        Sets up the training environment, creates necessary directories,
        initializes logging systems, and collects system metadata.

        Args:
            config: Configuration dictionary with training settings.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wwandb = config["wandb"]["wwandb"]

        # Setup run directory structure
        base_dir = config.get("tensorboard_dir", f"runs/{config['model']['env_name']}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.get("run_name", "run")
        group_name = config.get("group_name", "group")
        exp_codename = config.get("experiment_codename", "exp")
        self.run_dir = f"{base_dir}/{group_name}/{exp_codename}/{run_name}_{timestamp}"

        # Initialize wandb if enabled
        if self.wwandb:
            self.wandb_run = wandb.init(
                project=config['wandb']['project_name'],
                name=f"{run_name}_{timestamp}",
                group=group_name,
                config=config,
                save_code=True,
                reinit=True
            )
        
        # Setup checkpoint directory
        self.ckpt_dir = f"{self.run_dir}/checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Initialize best metric tracking
        self.best_metric_name = config.get("online_inference", {}).get("best_checkpoint_metric")
        self.best_metric_value = -float('inf') if self.best_metric_name is not None else None
        self.best_checkpoint_path = self.run_dir

        # Collect system and environment metadata
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

        # Save configuration
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.run_dir)
        
    def log(self, metrics: dict, step: int = None) -> None:
        """Log metrics to both TensorBoard and wandb (if enabled).

        This method handles logging of training metrics to both TensorBoard
        and wandb (if enabled). It also manages best checkpoint tracking
        based on the specified metric.

        Args:
            metrics: Dictionary of metrics to log.
                Keys are metric names, values can be:
                - int or float: Scalar metrics
                - torch.Tensor: Tensor metrics (will be converted to scalar)
            step: Current training step (optional).
                If not provided, uses internal global_step counter.

        Notes:
            - Automatically increments global_step if step is not provided
            - Checks for new best metric value if best_metric_name is set
            - Saves checkpoint when new best metric value is found
            - Handles both scalar and tensor metrics
            - Prints metrics to console for monitoring
        """
        if step is None:
            step = getattr(self, 'global_step', 0)
            if hasattr(self, 'global_step'):
                self.global_step += 1

        print(f"metrics: {metrics}")
        # Check for new best metric value
        if self.best_metric_name is not None and self.best_metric_name in metrics:
            current_value = metrics[self.best_metric_name]
            if current_value > self.best_metric_value:
                self.best_metric_value = current_value
                if hasattr(self, 'model'):  # Make sure model exists
                    self.save_checkpoint(is_best=True)

        # Log to wandb if enabled
        if self.wwandb:
            wandb.log(metrics)
            
        # Log to TensorBoard
        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
                elif isinstance(value, torch.Tensor):
                    self.writer.add_scalar(key, value.item(), step)

    def cleanup(self) -> None:
        """Clean up resources and finalize logging.

        This method should be called at the end of training to properly
        close TensorBoard writer and wandb run (if enabled).

        Notes:
            - Closes TensorBoard writer
            - Finalizes wandb run if enabled
            - Should be called before program exit
        """
        if hasattr(self, 'writer'):
            self.writer.close()
        if self.wwandb and hasattr(self, 'wandb_run'):
            self.wandb_run.finish()