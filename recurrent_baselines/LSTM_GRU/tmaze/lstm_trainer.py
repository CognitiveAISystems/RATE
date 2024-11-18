import torch
import wandb
from tqdm import tqdm
import numpy as np

from recurrent_baselines.LSTM_GRU import decision_lstm
from recurrent_baselines.LSTM_GRU.tmaze import val_tmaze
from TMaze_new.TMaze_new_src.utils import seeds_list
import torch.nn as nn

torch.backends.cudnn.benchmark = True

def train(model, optimizer, scheduler, raw_model, new_segment, epochs_counter, segments_count, wandb_step, ckpt_path, config, train_dataloader, val_dataloader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        model = decision_lstm.DecisionLSTM(**config['model_config'])

        wandb_step  = 0
        epochs_counter = 0
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training_config"]["learning_rate"], 
                                      weight_decay=config["training_config"]["weight_decay"], 
                                      betas=(config["training_config"]["beta_1"], config["training_config"]["beta_2"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/config["training_config"]["warmup_steps"], 1))
       
        raw_model = model.module if hasattr(model, "module") else model

        print(config)
        
    model.to(device)
    model.train()
    
    wwandb = config["wandb_config"]["wwandb"]
    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    it_counter = 0
    best_val_loss = np.inf
    epochs_without_improvement = 0
    max_epochs_without_improvement = 50

    switch = False
    val_loss = torch.tensor([float('inf')])
    
    pbar = tqdm(range(config["training_config"]["epochs"]))

    criterion_all = nn.CrossEntropyLoss(ignore_index=-10, reduction='mean')
    suc_rate = -1.0

    for epoch in pbar:
        is_train = True
        model.train()
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch

            x1 = s[:, :, :].to(device)
            y1 = a[:, :, :].to(device).long()
            r1 = rtg[:,:,:][:, :, :].to(device).float()

            
            with torch.set_grad_enabled(is_train):
                model.training = True
                logits = model(x1, y1, None, r1, None, None, wwandb=wwandb)
                
                logits = logits.reshape(-1, logits.size(-1))
                target = y1.reshape(-1).long()
                train_loss = criterion_all(logits, target)

                if wwandb:
                    wandb.log({"train_loss": train_loss.item()})

            if is_train:
                model.zero_grad()
                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                if config["training_config"]["grad_norm_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_config"]["grad_norm_clip"])
                optimizer.step()
                scheduler.step()
                lr = optimizer.state_dict()['param_groups'][0]['lr']

                pbar.set_description(f"[train] ep {epoch+1} it {it} SR {suc_rate:.2f} tTotal {train_loss.item():.2f} vTotal {val_loss.item():.2f} lr {lr:e}")
        it_counter += 1 
        epochs_counter += 1
        
        # Val
        model.eval()
        model.training = False
        is_train = False
        with torch.no_grad():
            for it, batch in enumerate(val_dataloader):        
                s, a, rtg, d, timesteps, masks = batch
                
                x1 = s[:, :, :].to(device)
                y1 = a[:, :, :].to(device).long()
                r1 = rtg[:,:,:][:, :, :].to(device).float() 
            
                with torch.set_grad_enabled(is_train):
                    optimizer.zero_grad()
                    logits = model(x1, y1, None, r1, None, None)

                    logits = logits.reshape(-1, logits.size(-1))
                    target = y1.reshape(-1).long()
                    val_loss = criterion_all(logits, target)


                    if wwandb:
                        wandb.log({"val_loss": val_loss.item()})
            
                pbar.set_description(f"[val] ep {epoch+1} it {it} SR {suc_rate:.2f} tTotal {train_loss.item():.2f} vTotal {val_loss.item():.2f} lr {lr:e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement and config["training_config"]["use_erl_stop"] == True:
            print("Early stopping!")
            break        

        # Scheduler changer
        if it_counter >= config["training_config"]["warmup_steps"] and switch == False: # !
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, 
                                                          end_factor=config['training_config']['lr_end_factor'], 
                                                          total_iters=len(train_dataloader)*config["training_config"]["epochs"]*config["training_config"]["max_segments"])
            switch = True
        
        if wwandb:
            wandb.log({"segments_count": segments_count})
        
        if epoch == config["training_config"]["epochs"] - 1:
            if config["training_config"]["online_inference"]:
                model.eval()
                with torch.no_grad():
                    goods, bads = 0, 0
                    timers = []
                    rewards = []
                    seeds = seeds_list
                    pbar2 = range(len(seeds))
                    for indx, iii in enumerate(pbar2):
                        episode_return, act_list, t, _ , delta_t, attn_map = val_tmaze.get_returns_TMaze(model=model, ret=config["data_config"]["desired_reward"], 
                                                                                               seed=seeds[iii], 
                                                                                               episode_timeout=config["online_inference_config"]["episode_timeout"],
                                                                                               corridor_length=config["online_inference_config"]["corridor_length"], 
                                                                                               context_length=config["training_config"]["context_length"],
                                                                                               device=device, act_dim=1,
                                                                                               config=config, create_video=False)
                        if episode_return == config["data_config"]["desired_reward"]:
                            goods += 1
                        else:
                            bads += 1
                        timers.append(delta_t)
                        rewards.append(episode_return)
                        
                        pbar.set_description(f"[inference | {indx+1}/{len(seeds)}] ep {epoch+1} it {it} tTotal {train_loss.item():.2f} vTotal {val_loss.item():.2f} lr {lr:e}")
                            
                    suc_rate = goods / (goods + bads)
                    ep_time = np.mean(timers)

                    if wwandb:
                        wandb.log({"Success_rate": suc_rate, "Mean_D[time]": ep_time})
        
            model.train()
            wandb_step += 1 
            if wwandb:
                wandb.log({"checkpoint_step": wandb_step})
            torch.save(model.state_dict(), ckpt_path + '_save' + '_KTD.pth')

    return model, wandb_step, optimizer, scheduler, raw_model, epochs_counter