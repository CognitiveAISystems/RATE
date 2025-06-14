import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import wandb
import time
from tqdm import tqdm

from RATE import RATE_model

from src.base_trainer import BaseTrainer
from src.inference_handler import InferenceHandler
from src.utils.lr_scheduler import LearningRateScheduler
from src.utils.colorize_dict import print_config

from offline_rl_baselines.BC import BehaviorCloning
from offline_rl_baselines.CQL import ConservativeQLearning
from offline_rl_baselines.IQL import ImplicitQLearning
from offline_rl_baselines.DLSTM import DecisionLSTM
from offline_rl_baselines.LSDT import LongShortDecisionTransformer


class Trainer(BaseTrainer):
    """Trainer class for model training with support for multiple model architectures and environments.

    This class extends BaseTrainer to provide specialized training functionality for different
    model architectures (RATE, DT, BC, CQL, IQL, etc.) and environments (ViZDoom, TMaze,
    Minigrid Memory, etc.). It handles model initialization, training loops, loss calculation,
    checkpointing, and online inference.

    Attributes:
        config (dict): Configuration dictionary containing all training parameters.
        device (torch.device): Device to run training on (cuda/cpu).
        wwandb (bool): Whether wandb logging is enabled.
        model (nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        lr_scheduler (LearningRateScheduler): Custom learning rate scheduler.
        raw_model (nn.Module): Unwrapped model (without DataParallel).
        wandb_step (int): Current wandb logging step.
        global_step (int): Global training step counter.
        warmup_changed_to_decay (bool): Whether warmup phase has transitioned to decay.
        log_last_segment_loss_only (bool): Whether to log only last segment loss.
        use_cosine_decay (bool): Whether to use cosine learning rate decay.
        env_name (str): Name of the environment being used.
        ckpt_epoch (int): Interval between checkpoints in epochs.
        video_path (str): Path to save inference videos.
        current_metric_value (float): Current value of tracked metric.
        env: Environment instance for online inference.
        hidden: Hidden state for recurrent models.
        EFFECTIVE_SIZE_BLOCKS (int): Total sequence length (context_length * sections).
        BLOCKS_CONTEXT (int): Base context length for sequence processing.

    Notes:
        - Supports multiple model architectures: RATE, DT, RMT, TrXL, DTXL, BC, CQL, IQL, DLSTM, DMamba, LSDT
        - Handles different environments: ViZDoom, TMaze, Minigrid Memory, Memory Maze, POPGym, Mikasa Robo
        - Implements both warmup and decay learning rate schedules
        - Supports online inference during training
        - Manages model checkpoints and best model selection
    """

    def __init__(self, config):
        """Initialize the trainer with configuration and setup training environment.

        Args:
            config (dict): Configuration dictionary containing all training parameters.
                Required keys:
                - wandb.wwandb: Whether to use wandb logging
                - training.log_last_segment_loss_only: Whether to log only last segment loss
                - training.use_cosine_decay: Whether to use cosine learning rate decay
                - model.env_name: Name of the environment
                - training.ckpt_epoch: Checkpoint interval in epochs
                - training.context_length: Base context length
                - training.sections: Number of sequence sections
        """
        super().__init__(config)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wwandb = config["wandb"]["wwandb"]
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.lr_scheduler = None
        self.raw_model = None
        self.wandb_step = 0
        self.global_step = 0
        self.warmup_changed_to_decay = False
        self.log_last_segment_loss_only = config["training"]["log_last_segment_loss_only"]
        self.use_cosine_decay = config["training"]["use_cosine_decay"]
        self.env_name = config["model"]["env_name"]
        self.ckpt_epoch = config["training"]["ckpt_epoch"]
        self.video_path = None
        self.current_metric_value = None

        self.env = None
        self.hidden = None

        if self.env_name == 'tmaze':
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_tmaze
        elif self.env_name == 'vizdoom':
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_vizdoom
        elif self.env_name == 'minigrid_memory':
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_minigridmemory
        elif self.env_name == 'memory_maze':
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_memorymaze
        elif 'popgym' in self.env_name:
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_popgym
        elif "mikasa_robo" in self.env_name:
            self._perform_mini_inference_impl = InferenceHandler.perform_mini_inference_mikasarobo

        self.EFFECTIVE_SIZE_BLOCKS = config["training"]["context_length"] * config["training"]["sections"]
        self.BLOCKS_CONTEXT = config["training"]["context_length"]

    def initialize_model(self):
        """Initialize the model, optimizer, and learning rate scheduler.

        This method:
        1. Clears CUDA cache if available
        2. Gets input dimensions from first batch
        3. Initializes appropriate model based on model_mode
        4. Sets up optimizer and learning rate scheduler
        5. Moves model to appropriate device
        6. Prints model parameters and configuration

        Notes:
            - For POPGym environments, automatically updates state_dim from data
            - Initializes model-specific parameters (e.g., biases for RATE)
            - Supports multiple model architectures with appropriate initialization
            - Updates configuration file with any auto-detected parameters
        """
        # Clear any CUDA cache before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get the first batch from the dataloader to get the state and action dimensions
        print('Pre-run:')
        pre_run = next(iter(self.train_dataloader))
        if "popgym" in self.env_name:
            self.config["model"]["state_dim"] = pre_run[0].shape[-1]
            
            # Update the config file with the new state_dim value
            config_path = os.path.join(self.run_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                saved_config["model"]["state_dim"] = self.config["model"]["state_dim"]
                with open(config_path, 'w') as f:
                    json.dump(saved_config, f, indent=4)
            print('state_dim', pre_run[0].shape[-1], pre_run[0].shape)
            print('act_dim', pre_run[1].shape[-1], pre_run[1].shape)

        if self.config["model_mode"] in ["RATE", "DT", "RMT", "TrXL", "DTXL"]:
            self.model = RATE_model.RATE(**self.config["model"])
            torch.nn.init.xavier_uniform_(self.model.r_w_bias)
            torch.nn.init.xavier_uniform_(self.model.r_r_bias)
        elif self.config["model_mode"] == "BC":
            self.model = BehaviorCloning(**self.config["model"])
        elif self.config["model_mode"] == "CQL":
            self.model = ConservativeQLearning(**self.config["model"])
        elif self.config["model_mode"] == "IQL":
            self.model = ImplicitQLearning(**self.config["model"])
        elif self.config["model_mode"] == "DLSTM":
            self.model = DecisionLSTM(**self.config["model"])
        elif self.config["model_mode"] == "DMamba":
            from offline_rl_baselines.DMamba import DMamba
            self.model = DMamba(**self.config["model"])
        elif self.config["model_mode"] == "LSDT":
            self.model = LongShortDecisionTransformer(**self.config["model"])
        else:
            raise ValueError(f"Invalid model type: {self.config['model_mode']}")

        self.lr_scheduler = LearningRateScheduler(self.config, self.train_dataloader)
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            betas=(
                self.config["training"]["beta_1"],
                self.config["training"]["beta_2"],
            )
        )
        
        if not self.use_cosine_decay:
            self.scheduler = self.lr_scheduler.make_warmup_scheduler(self.optimizer)

        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        
        print(f"Model parameters: {sum(p.numel() for p in list(self.model.parameters()))}")
        print("\nConfiguration:")
        print_config(self.config)
        print('\n')

    def calculate_losses(self, logits, target, masks, flag, q1_value=None, q2_value=None, cql_loss=None, v_value=None, iql_loss=None):
        """Calculate training losses and metrics for different environments and models.

        Args:
            logits (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.
            masks (torch.Tensor): Mask tensor for valid positions.
            flag (int): Whether this is the last segment (1) or not (0).
            q1_value (torch.Tensor, optional): Q1 values for CQL.
            q2_value (torch.Tensor, optional): Q2 values for CQL.
            cql_loss (torch.Tensor, optional): CQL loss component.
            v_value (torch.Tensor, optional): Value function output for IQL.
            iql_loss (dict, optional): IQL loss components.

        Returns:
            tuple: (optimization_loss, additional_metrics)
                - optimization_loss (torch.Tensor): Total loss for optimization
                - additional_metrics (dict): Dictionary of metrics to log

        Notes:
            Environment-specific loss calculations:
            - TMaze: Cross-entropy loss with special handling for last token
            - ViZDoom: Standard cross-entropy loss
            - Minigrid Memory: Cross-entropy loss with padding mask
            - Memory Maze: Standard cross-entropy loss
            - POPGym: MSE loss for pendulum tasks, cross-entropy for others
            - Mikasa Robo: MSE loss

            Model-specific loss handling:
            - IQL: Combines BC loss with IQL components
            - CQL: Combines BC loss with CQL regularization
            - Others: Uses standard loss calculation
        """
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

        elif self.env_name == 'memory_maze':
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1).long()
            )

        elif 'popgym' in self.env_name:
            if any(char in self.env_name \
                   for char in [
                       'NoisyPositionOnlyPendulumEasy',
                       'NoisyPositionOnlyPendulumMedium',
                       'NoisyPositionOnlyPendulumHard',
                       'PositionOnlyPendulumEasy',
                       'PositionOnlyPendulumMedium',
                       'PositionOnlyPendulumHard',
                    ]):
                loss = F.mse_loss(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1, 1).float(),
                    reduction='none'
                )
                mask = (target.reshape(-1, 1) != -10).float()
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1).long(),
                    ignore_index=-10
                )
        elif "mikasa_robo" in self.env_name:
            loss = F.mse_loss(logits, target)

        additional_metrics = {}

        # Add IQL loss if available
        if iql_loss is not None:
            # IQL returns a dictionary of loss components
            for loss_name, loss_value in iql_loss.items():
                additional_metrics[f"iql_{loss_name}"] = loss_value
            
            # Use the total IQL loss for optimization
            optimization_loss = loss  # BC loss is already included in IQL's update
            additional_metrics["bc_loss"] = loss.item()
            additional_metrics["total_loss"] = optimization_loss.item()
        # Add CQL loss if available
        elif cql_loss is not None:
            additional_metrics["cql_loss"] = cql_loss
            print('cql_loss', cql_loss)
            # Combine BC loss with CQL loss
            optimization_loss = loss + cql_loss
            additional_metrics["bc_loss"] = loss.item()
            additional_metrics["total_loss"] = optimization_loss.item()
        else:
            optimization_loss = loss

        if metrics_to_log:
            for metric_name, metric_value in metrics_to_log.items():
                if metric_value is not None:
                    if isinstance(metric_value, torch.Tensor):
                        additional_metrics[metric_name] = metric_value.item()
                    else:
                        additional_metrics[metric_name] = metric_value
        
        return optimization_loss, additional_metrics

    def log(self, metrics, step=None):
        """Log metrics to both WandB and TensorBoard.

        Args:
            metrics (dict): Dictionary of metrics to log.
                Values can be:
                - int or float: Scalar metrics
                - torch.Tensor: Tensor metrics (will be converted to scalar)
            step (int, optional): Global step for logging.
                If None, uses internal global_step counter.

        Notes:
            - Automatically increments global_step if step is not provided
            - Handles both scalar and tensor metrics
            - Skips None values
            - Logs to both WandB and TensorBoard if enabled
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
        """Perform online inference during training.

        Args:
            episode_timeout (int): Maximum number of timesteps per episode.
            text (str): Description text for the inference run.
            **env_specific_args: Environment-specific arguments.
                For TMaze:
                    - corridor_length (int): Length of the corridor
                For other environments:
                    - No additional arguments required

        Returns:
            dict: Inference results including:
                - success_rate (float): Success rate of the inference
                - episode_returns (list): List of episode returns
                - video_path (str, optional): Path to saved inference video

        Notes:
            - Uses environment-specific inference implementations
            - Handles different environment requirements
            - Supports video recording for supported environments
            - Manages environment lifecycle (creation/cleanup)
        """
        if self.env_name == 'tmaze':
            return self._perform_mini_inference_impl(
                self,  # passing self as first argument
                episode_timeout=episode_timeout,
                corridor_length=env_specific_args['corridor_length'],
                text=text, env=self.env
            )
        elif any(char in self.env_name for char in ('vizdoom', 'minigrid_memory', 'memory_maze', 'popgym', 'mikasa_robo')):
            return self._perform_mini_inference_impl(
                self,
                episode_timeout=episode_timeout,
                text=text, env=self.env
            )

    def log_metrics(self, loss, additional_metrics, flag, log_last_segment_only=True):
        """Log training metrics to wandb.

        Args:
            loss (torch.Tensor): Main training loss value.
            additional_metrics (dict): Dictionary of additional metrics to log.
            flag (int): Whether this is the last segment (1) or not (0).
            log_last_segment_only (bool): Whether to log metrics only for last segment.

        Notes:
            - Only logs if wandb is enabled
            - Filters out None values from metrics
            - Converts tensor metrics to scalar values
            - Respects log_last_segment_only setting
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

    def _should_checkpoint(self, epoch, tokens):
        """Determine if a checkpoint should be saved.

        Args:
            epoch (int): Current training epoch.
            tokens (int): Number of processed tokens.

        Returns:
            bool: True if checkpoint should be saved, False otherwise.

        Notes:
            Checkpoints are saved when:
            - Current epoch is a multiple of ckpt_epoch
            - Current epoch is the last epoch
            - Current epoch is 0 (initial checkpoint)
        """
        return ((epoch + 1) % int(self.config["training"]["ckpt_epoch"])) == 0 or epoch == self.config["training"]["epochs"] - 1 or epoch == 0

    def train(self, train_dataloader):
        """Main training loop.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.

        Returns:
            nn.Module: Trained model.

        Training Pipeline:
            1. Model Initialization:
               - Initialize model if not already done
               - Set model to training mode
               - Initialize training state

            2. Training Loop:
               For each epoch:
               - Process batches in segments
               - Calculate losses and metrics
               - Update model parameters
               - Update learning rate
               - Log metrics
               - Perform online inference at checkpoints

            3. Learning Rate Management:
               - Warmup phase
               - Transition to decay phase
               - Cosine or linear decay

            4. Checkpointing:
               - Save regular checkpoints
               - Save best model based on metric
               - Perform online inference

        Notes:
            - Supports multiple model architectures
            - Handles different environment requirements
            - Manages learning rate scheduling
            - Performs online inference at checkpoints
            - Tracks and logs various metrics
            - Supports both warmup and decay phases
        """
        self.train_dataloader = train_dataloader
        
        if self.model is None:
            self.initialize_model()
            
        self.model.train()
        it_counter = 0
        tokens = 0
        self.pbar = tqdm(range(self.config["training"]["epochs"]))
        printed = False

        for epoch in self.pbar:
            epoch_start_time = time.time()
            self.epoch = epoch
            is_train = True
            self.model.train()

            # Reset hidden state in the beginning of each epoch
            if hasattr(self.model, 'backbone') and self.model.backbone in ['lstm', 'gru']:
                self.hidden = None

            not_ep = False
            not_b = False
            
            for it, batch in enumerate(train_dataloader):
                s, a, rtg, d, timesteps, masks = batch
                if not not_ep and not printed:
                    print('Input shape (obs) before segmenting:', s.shape)
                    not_ep = True
                memory = None
                mem_tokens = None

                # Reset hidden state for each new batch if specified in the config
                if hasattr(self.model, 'backbone') and self.model.backbone in ['lstm', 'gru']:
                    if self.config.get("reset_hidden_state_batch", True):
                        self.hidden = self.model.reset_hidden(s.size(0), self.device)
                
                block_part_range = range(self.EFFECTIVE_SIZE_BLOCKS // self.BLOCKS_CONTEXT)
                
                for block_part in block_part_range:
                    from_idx = block_part * self.BLOCKS_CONTEXT
                    to_idx = (block_part + 1) * self.BLOCKS_CONTEXT

                    x1 = s[:, from_idx:to_idx, :].to(self.device)
                    y1 = a[:, from_idx:to_idx, :].to(self.device).float()
                    r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(self.device).float() 
                    t1 = timesteps[:, from_idx:to_idx].to(self.device)
                    masks1 = masks[:, from_idx:to_idx].to(self.device)
                    if not not_b and not printed:
                        print('Input shape (obs) after segmenting: ', x1.shape)
                        not_b = True
                        printed = True


                    # print the index of the tensor where the first element is not equal to 0
                    # print(block_part, y1[0])
                    # print(block_part, torch.where(y1[:] != 0)[0][0].item())

                    flag = 1 if block_part == max(block_part_range) else 0
                    
                    if mem_tokens is not None:
                        mem_tokens = mem_tokens.detach()
                    elif self.raw_model.mem_tokens is not None:
                        mem_tokens = self.raw_model.mem_tokens.repeat(1, r1.shape[0], 1)

                    with torch.set_grad_enabled(is_train):
                        # Modify forward call to support LSTM
                        if hasattr(self.model, 'backbone') and self.model.backbone in ['lstm', 'gru']:
                            res = self.model(
                                x1, y1, r1, y1, t1,
                                hidden=self.hidden,
                                mem_tokens=mem_tokens,
                                masks=masks1
                            )
                            self.hidden = res.get('hidden', None)
                            # Detach hidden state from the computation graph
                            if self.hidden is not None:
                                self.hidden = tuple(h.detach() for h in self.hidden)
                        else:
                            # Standard call for other models
                            res = self.model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None \
                                else self.model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                        
                        logits = res['logits']
                        memory = res.get('new_mems', None)
                        mem_tokens = res.get('mem_tokens', None)

                        # For IQL, we need to handle the case differently
                        if self.config["model_mode"] == "IQL":
                            # For IQL, we need next states for the value function update
                            # We'll use the next states in the sequence
                            if block_part < max(block_part_range):
                                next_from_idx = (block_part + 1) * self.BLOCKS_CONTEXT
                                next_to_idx = (block_part + 2) * self.BLOCKS_CONTEXT
                                next_x = s[:, next_from_idx:next_to_idx, :].to(self.device)
                            else:
                                # For the last block, we'll just use the last state
                                next_x = x1[:, -1:, :]
                            
                            # Extract rewards and terminals from the batch
                            rewards = r1[:, :, 0]  # Assuming rewards are in the first channel
                            terminals = d[:, from_idx:to_idx].to(self.device).float()
                            
                            # Call IQL update method
                            if hasattr(self.model, 'update'):
                                iql_loss = self.model.update(
                                    observations=x1,
                                    actions=y1,
                                    next_observations=next_x,
                                    rewards=rewards,
                                    terminals=terminals
                                )
                                
                                # Add IQL loss to the results
                                res['iql_loss'] = iql_loss
                        
                        # Extract Q-values and CQL loss if available (for CQL model)
                        q1_value = res.get('q1_value', None)
                        q2_value = res.get('q2_value', None)
                        cql_loss = res.get('cql_loss', None)
                        
                        # Extract IQL specific values if available
                        v_value = res.get('v_value', None)
                        iql_loss = res.get('iql_loss', None)
                        
                        loss, additional_metrics = self.calculate_losses(
                            logits, target=y1, masks=masks1, flag=flag,
                            q1_value=q1_value, q2_value=q2_value, cql_loss=cql_loss,
                            v_value=v_value, iql_loss=iql_loss
                        )

                        if self.wwandb:
                            self.log_metrics(
                                loss=loss, additional_metrics=additional_metrics, 
                                flag=flag, log_last_segment_only=self.log_last_segment_loss_only
                            )

                    if is_train:
                        # For IQL, optimization is already done in the update method
                        if self.config["model_mode"] != "IQL":
                            self.optimizer.zero_grad()
                            loss.backward(retain_graph=True)
                            if self.config["training"]["grad_norm_clip"] is not None:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_norm_clip"])
                            self.optimizer.step()
                            if not self.use_cosine_decay:
                                self.scheduler.step()  
                        
                        tokens += (y1 >= 0).sum().item() # TODO: change to the padding_idx?
                        
                        if self.use_cosine_decay:
                            # Update learning rate
                            lr = self.lr_scheduler.update_learning_rate(self.optimizer, tokens)
                            self.log({"learning_rate": lr})
                        else:
                            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                        self.log({"learning_rate": lr})

                self.pbar.set_description(f"[train] ep {epoch+1} it {it} tTotal {loss.item():.2f} lr {lr:e} tokens, M {(tokens/1e6):.2f}")
                    
                it_counter += 1 
            
            if it_counter >= self.config["training"]["warmup_steps"] and not self.warmup_changed_to_decay and not self.use_cosine_decay:
                self.scheduler = self.lr_scheduler.make_decay_scheduler(self.optimizer)
                self.warmup_changed_to_decay = True

            # Calculate and log epoch time
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            if self.wwandb:
                self.log({"epoch_time": epoch_time})
            
            # * Mini-inference at checkpoint
            if self._should_checkpoint(epoch, tokens):
                if self.config["training"]["online_inference"]:
                    if self.env_name == 'tmaze':
                        self.perform_mini_inference(
                            episode_timeout=self.config["online_inference"]["episode_timeout"],
                            corridor_length=self.config["online_inference"]["corridor_length"],
                            text=None, env=self.env
                        )

                    # Run multiple timeouts evaluation after the final epoch
                    if epoch == self.config["training"]["epochs"] - 1 and "multiple_timeouts" in self.config["online_inference"]:
                        print("\n\033[1;92mRunning final evaluation with multiple timeouts...\033[0m")
                        for timeout in self.config["online_inference"]["multiple_timeouts"]:
                            print(f"\n\033[1;93mEvaluating with timeout: {timeout}\033[0m")
                            self.perform_mini_inference(
                                episode_timeout=timeout,
                                corridor_length=timeout-2,
                                text=f"timeout_{timeout}", env=self.env
                            )

                    elif any(char in self.env_name for char in ('vizdoom', 'minigrid_memory', 'memory_maze', 'popgym', 'mikasa_robo')):
                        if "mikasa_robo" in self.env_name:
                            # Due to the peculiarities of GPU usage in ManiSkill3, we have to initialize
                            # the inference function in a separate way
                            from src.envs.mikasa_robo.mikasa_robo_initialization import InitializeMikasaRoboEnv
                            self.env = InitializeMikasaRoboEnv.create_mikasa_robo_env(
                                self.config["model"]["env_name"], self.run_dir, self.config
                            )

                        self.perform_mini_inference(
                            episode_timeout=self.config["online_inference"]["episode_timeout"],
                            text=None, env=self.env
                        )

                        if "mikasa_robo" in self.env_name and self.env is not None:
                            self.env.close()
                            self.env = None
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
        
                self.save_checkpoint()       

                self.wandb_step += 1 
                if self.wwandb:
                    self.log({"checkpoint_step": self.wandb_step})

        return self.model
    
    def save_checkpoint(self):
        """Save model checkpoint and update best model if necessary.

        This method:
        1. Saves current model state
        2. Updates best model if current metric is better
        3. Manages checkpoint paths and naming

        Notes:
            - Saves checkpoints in ckpt_dir
            - Updates best model in best_checkpoint_path
            - Handles both regular and best checkpoints
            - Preserves model training state
        """
        self.model.eval()
        save_path = os.path.join(self.ckpt_dir, f'step_{self.wandb_step}.pth')
        torch.save(self.model.state_dict(), save_path)
        if self.current_metric_value is not None and self.current_metric_value > self.best_metric_value:
            self.best_metric_value = self.current_metric_value
            save_path = os.path.join(self.best_checkpoint_path, f'best_checkpoint.pth') 
            torch.save(self.model.state_dict(), save_path)
        self.model.train()

    def __enter__(self):
        """Context manager entry.

        Returns:
            Trainer: Self instance.
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit.

        Args:
            exc_type: Exception type if any.
            exc_val: Exception value if any.
            exc_tb: Exception traceback if any.

        Notes:
            - Calls cleanup on exit
            - Handles any exceptions
            - Ensures proper resource cleanup
        """
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources and finalize training.

        This method:
        1. Closes TensorBoard writer
        2. Closes environment if open
        3. Clears CUDA cache
        4. Performs any necessary cleanup

        Notes:
            - Should be called at the end of training
            - Handles resource cleanup
            - Manages environment lifecycle
            - Cleans up GPU memory
        """
        if hasattr(self, 'writer'):
            self.writer.close()
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
            self.env = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def close(self):
        """Close TensorBoard writer and perform final cleanup.

        Notes:
            - Called by __del__
            - Ensures proper resource cleanup
            - Closes TensorBoard writer
        """
        if hasattr(self, 'writer'):
            self.writer.close()
            
    def __del__(self):
        """Destructor.

        Notes:
            - Calls close method
            - Ensures proper cleanup
            - Handles object destruction
        """
        self.close()