import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from RATE import mem_transformer_v2_GTrXL

from TMaze_new.TMaze_new_src.utils import seeds_list
from pprint import pprint
import math

# T-Maze:
from TMaze_new.TMaze_new_src.inference.val_tmaze import get_returns_TMaze

# VizDoom:
from VizDoom.VizDoom_src.inference.val_vizdoom import get_returns_VizDoom


from .base_trainer import BaseTrainer
from .inference_handler import InferenceHandler




class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wwandb = config["wandb_config"]["wwandb"]
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.raw_model = None
        self.epochs_counter = 0
        self.wandb_step = 0
        self.warmup_changed_to_decay = False
        # self.ckpt_dir = config["ckpt_dir"]
        self.log_last_segment_loss_only = config["training_config"]["log_last_segment_loss_only"]
        self.use_cosine_decay = config["training_config"]["use_cosine_decay"]
        self.env_name = config["model_config"]["mode"]

        if self.env_name == 'tmaze':
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_tmaze
        elif self.env_name == 'vizdoom':
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_vizdoom
        elif self.env_name == 'minigrid_memory':
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_minigridmemory
        
        # Constants
        self.EFFECTIVE_SIZE_BLOCKS = config["training_config"]["context_length"] * config["training_config"]["sections"]
        self.BLOCKS_CONTEXT = config["training_config"]["context_length"]

        # Initialize loggers
        # self.writer = SummaryWriter(log_dir=config.get("tensorboard_dir", "runs/experiment"))
        self.global_step = 0

    def initialize_model(self):
        self.model = mem_transformer_v2_GTrXL.MemTransformerLM(**self.config["model_config"])
        torch.nn.init.xavier_uniform_(self.model.r_w_bias)
        torch.nn.init.xavier_uniform_(self.model.r_r_bias)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training_config"]["learning_rate"],
            weight_decay=self.config["training_config"]["weight_decay"],
            betas=(
                self.config["training_config"]["beta_1"],
                self.config["training_config"]["beta_2"],
            )
        )
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps+1)/self.config["training_config"]["warmup_steps"], 1))
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.model.to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in list(self.model.parameters()))}")
        
        print("\nConfiguration:")
        pprint(self.config, indent=2, width=80)
        print("\n")

    def make_decay_scheduler(self, optimizer):
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=self.config['training_config']['lr_end_factor'], 
            total_iters=len(self.train_dataloader) * self.config["training_config"]["epochs"] * self.config["training_config"]["max_segments"])
        
        return decay_scheduler

    def calculate_losses(self, logits, target, masks, flag):
        metrics_to_log = {}

        if self.env_name == 'tmaze':
            # select targets for the last loss
            if flag == 1:
                logits_last = torch.zeros((logits.shape[0], 1, 4))
                target_last = torch.zeros((target.shape[0], 1, 1))
                for batch_num in range(logits.shape[0]):
                    ind = torch.where(target[batch_num].squeeze()==-10)[0][0].item() - 1
                    logits_last[batch_num] = logits[batch_num, ind]
                    target_last[batch_num] = target[batch_num, ind]

                # calculate train success rate
                train_sr = 0
                with torch.no_grad():
                    for tr_batch_num in range(target.shape[0]):
                        y_real = target[tr_batch_num].squeeze()
                        mask_real = masks[tr_batch_num]
                        act_real = torch.sum(y_real * mask_real)
                        y_pred = torch.argmax(torch.softmax(logits[tr_batch_num].squeeze(), dim=-1), dim=-1)
                        act_pred = y_pred[torch.where(y_real != 0)[0][0].item()]
                        if act_pred == act_real:
                            train_sr += 1
                    last_acc = train_sr / target.shape[0]

            # calculate full accuracy
            probs = torch.softmax(logits, dim=-1)
            ans = torch.argmax(probs, dim=-1)
            for batch_num in range(target.shape[0]):
                if -10 in target[batch_num]:
                    ind = torch.where(target[batch_num]==-10)[0][0].item()
                    ans[batch_num, ind:] = -10
                    
            labels = target.squeeze(-1)
            accuracy = torch.mean(torch.eq(ans, labels).float())

            # calculate loss for the last important token
            if flag == 1:
                criterion_last = nn.CrossEntropyLoss(ignore_index=-10)
                logits_last = logits_last.reshape(-1, logits_last.shape[-1])
                target_last = target_last.reshape(-1).long()
                loss_last = criterion_last(logits_last, target_last)

            # calculate full loss for the optimization
            criterion_all = nn.CrossEntropyLoss(ignore_index=-10, reduction='mean')
            logits = logits.reshape(-1, logits.size(-1))
            target = target.reshape(-1).long()
            loss = criterion_all(logits, target)

            metrics_to_log = {
                "train_loss_last": loss_last if flag == 1 else None,
                "train_accuracy": accuracy if flag == 1 else None,
                "train_last_acc": last_acc if flag == 1 else None
            }

        elif self.env_name == 'vizdoom':
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1).long()
            )
        elif self.env_name == 'minigrid_memory':
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1).long(), 
                ignore_index=-10
            )

        additional_metrics = {}

        if metrics_to_log:
            for metric_name, metric_value in metrics_to_log.items():
                if metric_value is not None:
                    if isinstance(metric_value, torch.Tensor):
                        additional_metrics[metric_name] = metric_value.item()
                    else:
                        additional_metrics[metric_name] = metric_value
        
        optimization_loss = loss
        
        return optimization_loss, additional_metrics

    def log(self, metrics, step=None):
        """Universal logging method for both WandB and TensorBoard
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Global step for tensorboard. If None, uses self.global_step
        """
        if step is None:
            step = self.global_step
            self.global_step += 1

        # Log to WandB
        if self.wwandb:
            wandb.log(metrics)
            
        # Log to TensorBoard
        for key, value in metrics.items():
            if value is not None:  # Skip None values
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
                elif isinstance(value, torch.Tensor):
                    self.writer.add_scalar(key, value.item(), step)

    def perform_mini_inference(self, episode_timeout, text, **env_specific_args):
        """Unified interface for mini inference across different environments.
        
        Args:
            episode_timeout: Required timeout parameter for all environments
            text: Required text parameter for all environments
            **env_specific_args: Environment-specific arguments
        """
        if self.env_name == 'tmaze':
            return self._perform_mini_inference_impl(
                self,  # passing self as first argument
                episode_timeout=episode_timeout,
                corridor_length=env_specific_args['corridor_length'],
                text=text
            )
        elif self.env_name == 'vizdoom':
            return self._perform_mini_inference_impl(
                self,  # passing self as first argument
                episode_timeout=episode_timeout,
                text=text
            )
        elif self.env_name == 'minigrid_memory':
            return self._perform_mini_inference_impl(
                self,  # passing self as first argument
                episode_timeout=episode_timeout,
                text=text
            )


    def log_metrics(self, loss, additional_metrics, flag, log_last_segment_only=True):
        """Helper function to log training metrics to wandb.
        
        Args:
            loss: Main training loss value
            additional_metrics: Dict of additional metrics to log
            flag: Whether this is the last segment
            log_last_segment_only: Whether to log metrics only for last segment
        """
        if not self.wwandb:
            return
            
        should_log = (not log_last_segment_only) or (log_last_segment_only and flag == 1)
        
        if should_log:
            # Always log main loss
            self.log({"train_loss": loss.item()})
            
            # Log additional metrics if available
            if additional_metrics:
                filtered_metrics = {
                    k: v.item() if hasattr(v, 'item') else v 
                    for k, v in additional_metrics.items()
                    if v is not None
                }
                self.log(filtered_metrics)


# ! next two functions are for cosine decay after linear warmup:
    def get_lr_multiplier(self, tokens):
        """Calculate learning rate multiplier based on warmup and decay schedule.
        
        Args:
            tokens (int): Current number of processed tokens
            
        Returns:
            float: Learning rate multiplier
        """
        if tokens < self.config["training_config"]["warmup_steps"]:
            # linear warmup
            lr_mult = float(tokens) / float(max(1, self.config["training_config"]["warmup_steps"]))
        else:
            # cosine learning rate decay
            progress = float(tokens - self.config["training_config"]["warmup_steps"]) / float(
                max(1, self.config["training_config"]["final_tokens"] - self.config["training_config"]["warmup_steps"]))
            lr_mult = max(self.config["training_config"]["lr_end_factor"], 
                         0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr_mult

    def update_learning_rate(self, tokens):
        """Update the learning rate based on the number of processed tokens.
        
        Args:
            tokens (int): Current number of processed tokens
            
        Returns:
            float: Current learning rate
        """
        lr_mult = self.get_lr_multiplier(tokens)
        lr = self.config["training_config"]["learning_rate"] * lr_mult
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr

    def train(self, train_dataloader):
        self.train_dataloader = train_dataloader
        
        if self.model is None:
            self.initialize_model()
            
        self.model.train()
        it_counter = 0
        tokens = 0
        self.pbar = tqdm(range(self.config["training_config"]["epochs"]))

        for epoch in self.pbar:
            self.epoch = epoch
            is_train = True
            self.model.train()
            
            for it, batch in enumerate(train_dataloader):
                s, a, rtg, d, timesteps, masks = batch
                memory = None
                mem_tokens = None
                
                block_part_range = range(self.EFFECTIVE_SIZE_BLOCKS // self.BLOCKS_CONTEXT)
                
                for block_part in block_part_range:
                    from_idx = block_part * self.BLOCKS_CONTEXT
                    to_idx = (block_part + 1) * self.BLOCKS_CONTEXT

                    x1 = s[:, from_idx:to_idx, :].to(self.device)
                    y1 = a[:, from_idx:to_idx, :].to(self.device).float()
                    r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(self.device).float() 
                    t1 = timesteps[:, from_idx:to_idx].to(self.device)
                    masks1 = masks[:, from_idx:to_idx].to(self.device)

                    flag = 1 if block_part == max(block_part_range) else 0
                    
                    if mem_tokens is not None:
                        mem_tokens = mem_tokens.detach()
                    elif self.raw_model.mem_tokens is not None:
                        mem_tokens = self.raw_model.mem_tokens.repeat(1, r1.shape[0], 1)

                    with torch.set_grad_enabled(is_train):
                        res = self.model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None \
                            else self.model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                        logits = res['logits']
                        memory = res['new_mems']
                        mem_tokens = res['mem_tokens']

                        loss, additional_metrics = self.calculate_losses(logits, target=y1, masks=masks1, flag=flag)

                        if self.wwandb:
                            self.log_metrics(
                                loss=loss, additional_metrics=additional_metrics, 
                                flag=flag, log_last_segment_only=self.log_last_segment_loss_only
                            )

                    if is_train:
                        self.optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        if self.config["training_config"]["grad_norm_clip"] is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training_config"]["grad_norm_clip"])
                        self.optimizer.step()
                        self.scheduler.step()  

                        tokens += (y1 >= 0).sum() # TODO: change to the padding_idx?

                        if self.use_cosine_decay:
                            # Update learning rate
                            lr = self.update_learning_rate(tokens)
                            self.log({"learning_rate": lr})
                        else:
                            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                        self.log({"learning_rate": lr})

                        self.pbar.set_description(f"[train] ep {epoch+1} it {it} tTotal {loss.item():.2f} lr {lr:e} tokens, M {(tokens/1e6):.2f}")
                        
                it_counter += 1 
                self.epochs_counter += 1

            if it_counter >= self.config["training_config"]["warmup_steps"] and not self.warmup_changed_to_decay:
                self.scheduler = self.make_decay_scheduler(self.optimizer)
                self.warmup_changed_to_decay = True
                
            # * Mini-inference at checkpoint
            if ((epoch + 1) % int(self.config["training_config"]["ckpt_epoch"])) == 0 or epoch == self.config["training_config"]["epochs"] - 1 or epoch == 0:
                if self.config["training_config"]["online_inference"]:
                    if self.env_name == 'tmaze':
                        self.perform_mini_inference(
                            episode_timeout=self.config["online_inference_config"]["episode_timeout"],
                            corridor_length=self.config["online_inference_config"]["corridor_length"],
                            text=None,
                        )
                    elif self.env_name == 'vizdoom':
                        self.perform_mini_inference(
                            episode_timeout=self.config["online_inference_config"]["episode_timeout"],
                            text=None,
                        )
                    elif self.env_name == 'minigrid_memory':
                       self.perform_mini_inference(
                            episode_timeout=self.config["online_inference_config"]["episode_timeout"],
                            text=None,
                        )
                
                self.save_checkpoint()

            # Inference at all lengths at last epoch
            if epoch == self.config["training_config"]["epochs"] - 1 and self.env_name == 'tmaze' and self.config["training_config"]["last_inference"]:
                for _segments in [1, 2, 3, 5, 9, 16, 30]:
                    _episode_timeout = 30 * _segments
                    _corridor_length = 30 * _segments - 2
                    self.perform_mini_inference(
                        episode_timeout=_episode_timeout,
                        corridor_length=_corridor_length,
                        text=str(_segments)
                    )
            
                self.model.train()
                self.wandb_step += 1 
                if self.wwandb:
                    self.log({"checkpoint_step": self.wandb_step})

                

        return self.model, self.wandb_step, self.optimizer, self.scheduler, self.raw_model, self.epochs_counter
    
    def save_checkpoint(self):
        self.model.eval()
        # torch.save(self.model.state_dict(), self.ckpt_dir + '/_save' + '_KTD.pth')
        save_path = os.path.join(self.ckpt_dir, f'model_step_{self.wandb_step}_KTD.pth')
        torch.save(self.model.state_dict(), save_path)
        self.model.train()

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Explicit cleanup method"""
        if hasattr(self, 'writer'):
            self.writer.close()
            
    def close(self):
        """Explicit cleanup method"""
        if hasattr(self, 'writer'):
            self.writer.close()
            
    def __del__(self):
        self.close()











