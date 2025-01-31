import datetime
import wandb
from torch.utils.data import random_split, DataLoader
import argparse
import yaml
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn.functional as F
import copy
import argparse

# python3 doom_cql.py --seed 1

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
# parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from TMaze_new.TMaze_new_src.utils import set_seed, get_intro, TMaze_data_generator, CombinedDataLoader
from TMaze_new.TMaze_new_src.train import train

import os
import sys

import pickle
from tqdm import tqdm
#import env_vizdoom2
import matplotlib.pyplot as plt
from itertools import count
import time
import random

from offline_rl_baselines.mlp_agent_cql_bc import DecisionMLP

# *========== SYTEM SETTINGS ============*

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["TORCH_USE_CUDA_DSA"] = "1" 

# *========== CONFIGS SETTINGS ============*

with open("wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']

with open("TMaze_new/TMaze_new_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# *========== AGENT SETTINGS ============*

agent = DecisionMLP(4, 1, 32, num_layers=2, mode='tmaze')
agent.train()
agent.to(agent.device)

config['training_config']['grad_norm_clip'] = None # !!!


optimizer = torch.optim.AdamW(
    agent.parameters(), 
    lr=config["training_config"]["learning_rate"],
    weight_decay=config["training_config"]["weight_decay"], 
    betas=(config["training_config"]["beta_1"],
           config["training_config"]["beta_2"]))

def create_args():
    parser = argparse.ArgumentParser(description='RATE VizDoom trainer') 
    parser.add_argument('--seed',           type=int, default=1,       help='')
    parser.add_argument('--exp-name',       type=str, default='tmaze', help='')
    parser.add_argument('--stacked-input',  action='store_true', help='')
    parser.add_argument('--loss-mode',      type=str, default='bc', help='bc, cql')
    parser.add_argument('--num-epochs',     type=int, default=1000, help='')
    parser.add_argument('--context-length', type=int, default=30, help='')
    parser.add_argument('--segments',       type=int, default=3, help='')
    return parser


# *========== START TRAINING ============*

if __name__ == '__main__':
    args = create_args().parse_args()
    set_seed(args.seed)
    run_name = f'{args.exp_name}_{args.loss_mode}_{args.seed}_stacked_{args.stacked_input}_context_{args.context_length}_segments_{args.segments}'
    run = wandb.init(project="RATE_TMAZE_CQL", name=run_name, config=args, save_code=True)

    max_length = args.segments * args.context_length
    print(f"{max_length=}")

    # *========== DATASET SETTINGS ============*

    TMaze_data_generator(
        max_segments=args.segments, # TODO segments
        multiplier=config["data_config"]["multiplier"], 
        hint_steps=config["data_config"]["hint_steps"],
        desired_reward=config["data_config"]["desired_reward"], 
        win_only=config["data_config"]["win_only"], 
        segment_length=args.context_length) # TODO context_length


    combined_dataloader = CombinedDataLoader(
        n_init=1, 
        n_final=args.segments, # TODO segments
        multiplier=config["data_config"]["multiplier"], 
        hint_steps=config["data_config"]["hint_steps"], 
        batch_size=config["training_config"]["batch_size"], 
        mode="", 
        cut_dataset=config["data_config"]["cut_dataset"], 
        one_mixed_dataset=True, 
        desired_reward=config["data_config"]["desired_reward"], 
        win_only=config["data_config"]["win_only"],
        segment_length=args.context_length) # TODO context_length

    # Split dataset into train and validation sets
    full_dataset = combined_dataloader.dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use DataLoader to load the datasets in parallel
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["training_config"]["batch_size"], 
        shuffle=True, 
        num_workers=4)
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config["training_config"]["batch_size"], 
        shuffle=True, 
        num_workers=4)
    
    print(f"Number of considered segments: {args.segments}, \
          dataset length: {len(combined_dataloader.dataset)}, \
            Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    print(f"{args=}\n")

    criterion_all = nn.CrossEntropyLoss(ignore_index=-10, reduction='mean')
    

    agent.train()

    CQL_ALPHA = 1.0
    DISCOUNT = 0.99
    TAU = 0.005  # Soft update parameter
    TARGET_UPDATE_FREQ = 10

    target_q1 = copy.deepcopy(agent.q1)
    target_q2 = copy.deepcopy(agent.q2)

    for param in target_q1.parameters():
        param.requires_grad = False
    for param in target_q2.parameters():
        param.requires_grad = False

    for epochs in tqdm(range(args.num_epochs), desc="Training epochs"):
        agent.train()
    
        for it, batch in tqdm(enumerate(train_dataloader), 
                            desc=f"Epoch {epochs}", 
                            total=len(train_dataloader),
                            leave=False):  # leave=False prevents multiple progress bars
            s, a, rtg, d, timesteps, masks = batch

            done_mask = (d == 1) | (d == 2)
            d = done_mask.float().unsqueeze(-1)
            last_values = d[:, -1:, :]
            d = torch.cat([d, last_values], dim=1) 

            d = d.cuda()

            s = s.cuda()
            a = a.cuda()
            rtg = rtg.cuda().float()

            action_preds, q1_pred, q2_pred, cql_loss = agent(s, a, rtg, stacked_input=args.stacked_input)

            with torch.no_grad():
                next_s = s[:, 1:]
                next_a = a[:, 1:]
                next_rtg = rtg[:, 1:]

                _, next_q1, next_q2, _ = agent(next_s, next_a, next_rtg, stacked_input=args.stacked_input)
                next_q = torch.min(next_q1, next_q2)
                # print(rtg.shape, d.shape, next_s.shape, next_a.shape, next_rtg.shape, next_q1.shape, next_q2.shape, next_q.shape)
                target_q = rtg[:, :-1] + DISCOUNT * (1 - d[:, :-1]) * next_q

            q1_loss = F.mse_loss(q1_pred[:, :-1], target_q)
            q2_loss = F.mse_loss(q2_pred[:, :-1], target_q)   

            action_preds = action_preds.reshape(-1, action_preds.size(-1))
            target_actions = a.reshape(-1).long()
            bc_loss = criterion_all(action_preds, target_actions)
            
            if args.loss_mode == 'cql':
                total_loss = q1_loss + q2_loss + cql_loss + bc_loss
            elif args.loss_mode == 'bc':
                total_loss = bc_loss
            else:
                raise ValueError(f"Loss mode {args.loss_mode} not supported")
        
            optimizer.zero_grad()
            total_loss.backward()
            if config["training_config"]["grad_norm_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), config["training_config"]["grad_norm_clip"])
            optimizer.step()
        
        if it % 1 == 0:
            with torch.no_grad():
                for param, target_param in zip(agent.q1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(agent.q2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        tqdm.write(f'Epochs: {epochs} It: {it} Train Loss: {total_loss.item():.4f} '
                    f'(BC: {bc_loss.item():.4f}, Q1: {q1_loss.item():.4f}, '
                    f'Q2: {q2_loss.item():.4f}, CQL: {cql_loss.item():.4f})')

        wandb.log({'BC':bc_loss.item()})
        wandb.log({'Q1':q1_loss.item()})
        wandb.log({'Q2':q2_loss.item()})
        wandb.log({'CQL':cql_loss.item()})


        if epochs%10==0:
            PATH = f'offline_rl_baselines/ckpt/tmaze_ckpt_mlp/{args.loss_mode}/{args.seed}/{run_name}'
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            torch.save(agent.state_dict(),f"{PATH}.ckpt")

    run.finish()