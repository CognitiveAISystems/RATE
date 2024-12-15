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

sys.path.append('../')
from VizDoom.VizDoom_src.utils import get_vizdoom_iter_dataset, ViZDoomIterDataset
from VizDoom.VizDoom_src.train import trainer
from TMaze_new.TMaze_new_src.utils import set_seed

import os
import sys

import pickle
from tqdm import tqdm
#import env_vizdoom2
import matplotlib.pyplot as plt
from itertools import count
import time
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["TORCH_USE_CUDA_DSA"] = "1" 

with open("../VizDoom/VizDoom_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config["training_config"]["batch_size"] = 128
max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]
print(f"{max_length=}")

from lstm_agent_cql import DecisionLSTM

agent = DecisionLSTM(4, 1, 128,mode='doom')

# PATH = f'DOOM_BC_270_sar'
# weights = torch.load(f"{PATH}.ckpt", map_location="cpu")

# agent.load_state_dict(weights, strict=True)
agent.train()
agent.to(agent.device)

optimizer = torch.optim.AdamW(agent.parameters(), lr=config["training_config"]["learning_rate"], 
                                      weight_decay=config["training_config"]["weight_decay"], 
                                      betas=(config["training_config"]["beta_1"], config["training_config"]["beta_2"]))

path_to_splitted_dataset = '../../../RATE/VizDoom/VizDoom_data/iterative_data/'
train_dataset = ViZDoomIterDataset(path_to_splitted_dataset, 
                                 gamma=config["data_config"]["gamma"], 
                                 max_length=max_length, 
                                 normalize=config["data_config"]["normalize"])

train_dataloader = DataLoader(train_dataset, 
                             batch_size=config["training_config"]["batch_size"],
                             shuffle=True, 
                             num_workers=8)

with open("../wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']

def create_args():
    parser = argparse.ArgumentParser(description='RATE VizDoom trainer') 
    parser.add_argument('--seed',           type=int, default=1,       help='')
    return parser

# os.environ['WANDB_API_KEY'] = 'WANDB_API_KEY'
if __name__ == '__main__':
    args = create_args().parse_args()
    SEED = args.seed
    set_seed(SEED)
    EXP_NAME = 'doom_cql_sar'
    run = wandb.init(project="RATE_DOOM_CQL", name=f'{EXP_NAME}_{SEED}')

    criterion_all = nn.CrossEntropyLoss(ignore_index=-10, reduction='mean')
    agent.train()

    CQL_ALPHA = 1.0
    DISCOUNT = 0.99
    #TARGET_UPDATE_FREQ = 10
    TAU = 0.005  # Soft update parameter
    TARGET_UPDATE_FREQ = 10

    target_q1 = copy.deepcopy(agent.q1)
    target_q2 = copy.deepcopy(agent.q2)

    for param in target_q1.parameters():
        param.requires_grad = False
    for param in target_q2.parameters():
        param.requires_grad = False

    for epochs in range(3600):
        
        agent.train()
        
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch
            d[d==2] = 1.
            d = 1-d
            d = d.unsqueeze(-1).cuda()
            s = s.cuda()
            a = a.cuda()
            rtg = rtg.cuda().float()

            agent.init_hidden(s.shape[0])

            action_preds, q1_pred, q2_pred, cql_loss = agent(s,a,rtg, stacked_input=True)

            with torch.no_grad():
                next_s = s[:, 1:]
                next_a = a[:, 1:]
                next_rtg = rtg[:, 1:]

                _, next_q1, next_q2, _ = agent(next_s, next_a, next_rtg, stacked_input=True)
                next_q = torch.min(next_q1, next_q2)
                target_q = rtg[:, :-1] + DISCOUNT * (1 - d[:, :-1]) * next_q

            q1_loss = F.mse_loss(q1_pred[:, :-1], target_q)
            q2_loss = F.mse_loss(q2_pred[:, :-1], target_q)   

        #     break
        # break
        
            action_preds = action_preds.reshape(-1, action_preds.size(-1))
            target_actions = a.reshape(-1).long()
            bc_loss = criterion_all(action_preds, target_actions)
            
            total_loss = q1_loss + q2_loss + cql_loss + bc_loss
            #total_loss = bc_loss
        
            optimizer.zero_grad()
            total_loss.backward()#retain_graph=False)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config["training_config"]["grad_norm_clip"])
            optimizer.step()
        
        if it % 1 == 0:
            with torch.no_grad():
                for param, target_param in zip(agent.q1.parameters(), target_q1.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(agent.q2.parameters(), target_q2.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        print(f'Epochs: {epochs} It: {it} Train Loss: {total_loss.item()} '
            f'(BC: {bc_loss.item():.4f}, Q1: {q1_loss.item():.4f}, '
            f'Q2: {q2_loss.item():.4f}, CQL: {cql_loss.item():.4f})')

        wandb.log({'BC':bc_loss.item()})
        wandb.log({'Q1':q1_loss.item()})
        wandb.log({'Q2':q2_loss.item()})
        wandb.log({'CQL':cql_loss.item()})


        if epochs%10==0:
            PATH = f'./ckpt/{SEED}/Doom_lstm_SAR_90_CQL'
            os.makedirs(PATH, exist_ok=True)
            torch.save(agent.state_dict(),f"{PATH}.ckpt")

    run.finish()