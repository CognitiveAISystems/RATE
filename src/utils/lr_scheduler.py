import math
import torch


class LearningRateScheduler:
    """Learning rate scheduler with warmup and decay strategies.

    This class implements a learning rate scheduling strategy that combines
    linear warmup and either linear or cosine decay. It supports both
    PyTorch's built-in schedulers and custom token-based scheduling.

    The scheduling process consists of two phases:
    1. Warmup: Linear increase from 0 to initial learning rate
    2. Decay: Either linear or cosine decay from initial to final learning rate

    Args:
        config: Configuration dictionary containing training parameters.
            Required keys:
            - training.warmup_steps: Number of steps for warmup (int)
            - training.learning_rate: Initial learning rate (float)
            - training.lr_end_factor: Final learning rate factor (float)
            - training.epochs: Number of training epochs (int)
            - training.max_segments: Maximum number of segments (int)
            - training.final_tokens: Total number of tokens for training (int)
        train_dataloader: PyTorch DataLoader for training data (optional).
            Used for calculating total training steps in linear decay.

    Attributes:
        config: Configuration dictionary containing training parameters.
        train_dataloader: Training data loader (optional).

    Notes:
        - The scheduler supports both PyTorch's built-in schedulers and
          custom token-based scheduling.
        - Warmup phase uses linear increase from 0 to initial learning rate.
        - Decay phase can use either linear or cosine decay.
        - For linear decay, the total steps are calculated based on
          dataloader length, epochs, and max segments.
        - For token-based scheduling, the decay is based on the number
          of processed tokens.
    """

    def __init__(self, config: dict, train_dataloader: torch.utils.data.DataLoader = None):
        """Initialize the learning rate scheduler.

        Args:
            config: Configuration dictionary with training parameters.
            train_dataloader: Optional DataLoader for training data.
        """
        self.config = config
        self.train_dataloader = train_dataloader

    def make_warmup_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        """Create a warmup scheduler using PyTorch's LambdaLR.

        Implements linear warmup from 0 to initial learning rate over
        warmup_steps steps.

        Args:
            optimizer: PyTorch optimizer to schedule.

        Returns:
            LambdaLR: PyTorch scheduler implementing linear warmup.

        Note:
            The learning rate multiplier is calculated as:
            lr_mult = min((steps + 1) / warmup_steps, 1)
        """
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda steps: min((steps+1)/self.config["training"]["warmup_steps"], 1)
        )
        return warmup_scheduler

    def make_decay_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LinearLR:
        """Create a linear decay scheduler using PyTorch's LinearLR.

        Implements linear decay from initial learning rate to
        initial_rate * lr_end_factor over the total training steps.

        Args:
            optimizer: PyTorch optimizer to schedule.

        Returns:
            LinearLR: PyTorch scheduler implementing linear decay.

        Note:
            Total steps are calculated as:
            total_steps = len(dataloader) * epochs * max_segments
        """
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=self.config['training']['lr_end_factor'], 
            total_iters=len(self.train_dataloader) * self.config["training"]["epochs"] * 
                        self.config["training"]["max_segments"]
        )
        return decay_scheduler
    
    def get_lr_multiplier(self, tokens: int) -> float:
        """Calculate learning rate multiplier based on warmup and decay schedule.
        
        Implements a combined warmup and cosine decay schedule based on
        the number of processed tokens. The schedule consists of:
        1. Linear warmup from 0 to 1 over warmup_steps
        2. Cosine decay from 1 to lr_end_factor over remaining steps

        Args:
            tokens: Current number of processed tokens.
            
        Returns:
            float: Learning rate multiplier in range [lr_end_factor, 1.0].

        Note:
            The multiplier is calculated as:
            - During warmup: tokens / warmup_steps
            - During decay: 0.5 * (1 + cos(π * progress))
            where progress = (tokens - warmup_steps) / (final_tokens - warmup_steps)
        """
        if tokens < self.config["training"]["warmup_steps"]:
            # linear warmup
            lr_mult = float(tokens) / float(max(1, self.config["training"]["warmup_steps"]))
        else:
            # cosine learning rate decay
            progress = float(tokens - self.config["training"]["warmup_steps"]) / float(
                max(1, self.config["training"]["final_tokens"] - self.config["training"]["warmup_steps"]))
            lr_mult = max(self.config["training"]["lr_end_factor"], 
                         0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr_mult

    def update_learning_rate(self, optimizer: torch.optim.Optimizer, tokens: int) -> float:
        """Update the learning rate based on the number of processed tokens.
        
        Updates the learning rate for all parameter groups in the optimizer
        based on the current number of processed tokens and the scheduling
        strategy.

        Args:
            optimizer: PyTorch optimizer to update.
            tokens: Current number of processed tokens.
            
        Returns:
            float: Current learning rate after update.

        Note:
            The learning rate is calculated as:
            lr = initial_lr * lr_multiplier
            where lr_multiplier is determined by get_lr_multiplier()
        """
        lr_mult = self.get_lr_multiplier(tokens)
        lr = self.config["training"]["learning_rate"] * lr_mult
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr