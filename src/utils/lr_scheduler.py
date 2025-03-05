import math
import torch

class LearningRateScheduler:
    def __init__(self, config, train_dataloader=None):
        self.config = config
        self.train_dataloader = train_dataloader

    def make_warmup_scheduler(self, optimizer):
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda steps: min((steps+1)/self.config["training"]["warmup_steps"], 1)
        )

        return warmup_scheduler

    def make_decay_scheduler(self, optimizer):
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=self.config['training']['lr_end_factor'], 
            total_iters=len(self.train_dataloader) * self.config["training"]["epochs"] * self.config["training"]["max_segments"])
        
        return decay_scheduler
    
    def get_lr_multiplier(self, tokens):
        """Calculate learning rate multiplier based on warmup and decay schedule.
        
        Args:
            tokens (int): Current number of processed tokens
            
        Returns:
            float: Learning rate multiplier
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

    def update_learning_rate(self, optimizer, tokens):
        """Update the learning rate based on the number of processed tokens.
        
        Args:
            tokens (int): Current number of processed tokens
            
        Returns:
            float: Current learning rate
        """
        lr_mult = self.get_lr_multiplier(tokens)
        lr = self.config["training"]["learning_rate"] * lr_mult
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr