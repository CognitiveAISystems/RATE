import numpy as np
import torch
import wandb
from tqdm import tqdm
import math
# import cv2
import time
import glob
import os
# import os
# import sys
# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
# parent_dir = os.path.dirname(parent_dir)
# sys.path.append(parent_dir)
# parent_dir = os.path.dirname(parent_dir)
# sys.path.append(parent_dir)

from RATE_GTrXL import mem_transformer_v2_GTrXL

# from VizDoom.VizDoom_src.utils import z_normalize, inverse_z_normalize

# from VizDoom.VizDoom_src.inference.val_vizdoom import get_returns_VizDoom
# from MemoryMaze.MemoryMaze_src.inference.val_mem_maze import get_returns_MemoryMaze 
# from MinigridMemory.MinigridMemory_src.inference.val_minigridmemory import get_returns_MinigridMemory

# import warnings
# warnings.filterwarnings('ignore')
import torch.nn as nn

reds = [2, 3, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20, 21, 25, 26, 27, 28, 29, 31, 38, 40, 41, 42, 45,
        46, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 61, 63, 64, 67, 68, 70, 72, 73, 74, 77, 80, 82, 84, 
        86, 88, 89, 90, 91, 92, 97, 98, 99, 100, 101, 103, 106, 108, 109, 113, 115, 116, 117, 120, 
        123, 124, 125, 126, 127, 128, 129, 133, 134, 136, 139, 140, 142, 144, 145, 147, 148, 151, 152, 
        153, 154, 156, 157, 158, 159, 161, 164, 165, 170, 171, 173]

greens = [0, 1, 4, 5, 7, 12, 13, 19, 22, 23, 24, 30, 32, 33, 34, 35, 36, 37, 39, 43, 44, 47, 48, 56, 57,
          62, 65, 66, 69, 71, 75, 76, 78, 79, 81, 83, 85, 87, 93, 94, 95, 96, 102, 104, 105, 107, 110, 111, 
          112, 114, 118, 119, 121, 122, 130, 131, 132, 135, 137, 138, 141, 143, 146, 149, 150, 155, 160, 162, 
          163, 166, 167, 168, 169, 172, 175, 176, 177, 182, 183, 187, 190, 192, 193, 195, 199, 204, 206, 208, 
          209, 210, 212, 215, 216, 218, 219, 220, 221, 223, 224, 225]


def train(ckpt_path, config, train_dataloader, mean, std, max_segments, experiment):
    if config['model_config']['mode'] == "memory_maze":
        from MemoryMaze.MemoryMaze_src.inference.val_mem_maze import get_returns_MemoryMaze 
        tokens_cnt = tokens_cnt_step= 8_000_000
    elif config['model_config']['mode'] == "doom":
        from VizDoom.VizDoom_src.inference.val_vizdoom import get_returns_VizDoom
    elif config['model_config']['mode'] == "minigrid_memory":
        from MinigridMemory.MinigridMemory_src.inference.val_minigridmemory import get_returns_MinigridMemory
    elif config['model_config']['mode'] == "maniskill-pushcube":
        from ManiSkill.ManiSkill_src.inference.val_maniskill import get_returns_ManiSkill
        from collections import defaultdict
        tokens_cnt = tokens_cnt_step = 2_000_000

        import gymnasium as gym
        import mani_skill.envs
        from mani_skill.utils import gym_utils
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
        from mani_skill.utils.wrappers.record import RecordEpisode
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        eval_output_dir = f"ManiSkill/val_videos/{config['model_mode']}/{config['text_description']}"
        env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos", render_mode="all", sim_backend="gpu")

        env = gym.make("PushCube-v1", num_envs=1, **env_kwargs) 
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env) 
        env = RecordEpisode(env, output_dir=eval_output_dir, save_trajectory=False, trajectory_name=f"rate_val", max_steps_per_video=50, video_fps=30)
        env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
    else:
        raise NotImplementedError


    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    episode_timeout = config["online_inference_config"]["episode_timeout"]
    use_argmax = config["online_inference_config"]["use_argmax"]

    MEAN = mean
    STD = std

    model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

    print(config)
    print(f"device: {device}")
    
    torch.nn.init.xavier_uniform_(model.r_w_bias);
    torch.nn.init.xavier_uniform_(model.r_r_bias);
    wandb_step  = 0

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config["training_config"]["learning_rate"], weight_decay=config["training_config"]["weight_decay"], 
                                  betas=(config["training_config"]["beta_1"], config["training_config"]["beta_2"]))

    raw_model = model.module if hasattr(model, "module") else model
        
    model.to(device)
    model.train()
    
    wwandb = config["wandb_config"]["wwandb"]
    wcomet = config['wandb_config']['wcomet']

    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    it_counter = 0
    switch = False

    EFFECTIVE_SIZE_BLOCKS = config["training_config"]["context_length"] * config["training_config"]["sections"]
    BLOCKS_CONTEXT = config["training_config"]["context_length"]
    
    pbar = tqdm(range(config["training_config"]["epochs"]))
    tokens_dict = {}
    tokens_dict[0] = None

    ckpt_dict = {}
    ckpt_dict[0] = None

    tokens = 0
    
    # tokens_cnt = 500_000

    for epoch in pbar:
        train_imgs = []
        is_train = True
        model.train()
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch

            if config["data_config"]["normalize"] == 2:
                _b, _l, _c, _h, _w = s.shape
                s = s.reshape(_b*_l, _c, _h, _w)
                s = z_normalize(s, MEAN, STD)
                s = s.reshape(_b, _l, _c, _h, _w)

            memory = None
            mem_tokens = None
            
            block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
            if tokens_dict[0] is None:
                for block_part in block_part_range:
                    tokens_dict[block_part] = None
            if ckpt_dict[0] is None:
                for block_part in block_part_range:
                    ckpt_dict[block_part] = None
                
            for block_part in block_part_range:
                from_idx = block_part*(BLOCKS_CONTEXT)
                to_idx = (block_part+1)*(BLOCKS_CONTEXT)
                
                x1 = s[:, from_idx:to_idx, :].to(device)
                y1 = a[:, from_idx:to_idx, :].to(device).float()
                r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(device).float() 
                t1 = timesteps[:, from_idx:to_idx].to(device)
                masks1 = masks[:, from_idx:to_idx].to(device)
                    
                model.flag = 1 if block_part == max(block_part_range) else 0
                if mem_tokens is not None:
                    mem_tokens = mem_tokens.detach()
                elif raw_model.mem_tokens is not None:
                    mem_tokens = raw_model.mem_tokens.repeat(1, r1.shape[0], 1)

                if config["model_config"]["num_mem_tokens"] > 0:
                    mean_tokens = torch.mean(mem_tokens, dim=1)[0]

                    if tokens_dict[block_part] is not None:
                        cos_sim = cos(mean_tokens, tokens_dict[block_part]).item()
                    else:
                        cos_sim = None

                with torch.set_grad_enabled(is_train):
                    optimizer.zero_grad()
                        
                    res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None \
                    else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                    memory = res[0][2:]
                    logits, train_loss = res[0][0], res[0][1]
                    mem_tokens = res[1]
                
                    if wwandb:
                        wandb.log({"train_loss":  train_loss.item()})
                    elif wcomet:
                        experiment.log_metric("train_loss", train_loss.item(), step=it_counter)

                    if config["model_config"]["num_mem_tokens"] > 0:
                        if wwandb:
                            wandb.log({f"cos_1st_token_block_{block_part}": cos_sim})
                            tokens_dict[block_part] = mean_tokens
                        elif wcomet:
                            experiment.log_metric(f"cos_1st_token_block_{block_part}", cos_sim, step=it_counter)
                            tokens_dict[block_part] = mean_tokens

                        if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or (epoch + 1) == 1:
                            if it == len(train_dataloader)-1:
                                if ckpt_dict[block_part] is not None:
                                    cos_sim_ckpt= cos(mean_tokens, ckpt_dict[block_part]).item()
                                else:
                                    cos_sim_ckpt = None

                                if wwandb:
                                    wandb.log({f"ckpt_cos_1st_token_block_{block_part}": cos_sim_ckpt})
                                elif wcomet:
                                    experiment.log_metric(f"ckpt_cos_1st_token_block_{block_part}", cos_sim_ckpt, step=it_counter)
                            

                                ckpt_dict[block_part] = mean_tokens

                if is_train:
                    model.zero_grad()
                    train_loss.backward(retain_graph=True)
                    if config["training_config"]["grad_norm_clip"] != 'None':
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_config"]["grad_norm_clip"])
                    optimizer.step()

                    # * decay the learning rate based on our progress
                    tokens += (y1 >= 0).sum()
                    if tokens < config["training_config"]["warmup_steps"]:
                        # linear warmup
                        lr_mult = float(tokens) / float(max(1, config["training_config"]["warmup_steps"]))
                    else:
                        # cosine learning rate decay
                        progress = float(tokens - config["training_config"]["warmup_steps"]) / float(max(1, config["training_config"]["final_tokens"] - config["training_config"]["warmup_steps"]))
                        lr_mult = max(config["training_config"]["lr_end_factor"], 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config["training_config"]["learning_rate"] * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr



                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    if wwandb:
                        wandb.log({"learning_rate": lr})
                    elif wcomet:
                        experiment.log_metric("learning_rate", lr, step=it_counter)

                it_counter += 1



            pbar.set_description(f"ep {epoch+1} it {it} tTotal {train_loss.item():.2f} lr {lr:e} tokens, M {(tokens/1e6):.2f}")

        if wwandb:
            wandb.log({"epochs": epoch+1})
        elif wcomet:
            experiment.log_metrics({"epochs": epoch+1}, step=it_counter)
        
        # ? Save
        if config["model_config"]["mode"] == 'doom':
            if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or (epoch + 1) == 1:
                if config["training_config"]["online_inference"] == True:
                    if config["model_config"]["mode"] == 'doom':
                        model.eval()
                        with torch.no_grad():
                            FRAME_SKIP = 2
                            def optimize_pillar(color, seeds, config, wwandb, wcomet):
                                for ret in [config["online_inference_config"]["desired_return_1"]]:
                                    returns = []
                                    ts = []
                                    attn_map_received = False
                                    for i in range(len(seeds)):
                                        episode_return, act_list, t, _, _, attn_map = get_returns_VizDoom(model=model, ret=ret, seed=seeds[i], 
                                                                                                episode_timeout=episode_timeout, 
                                                                                                context_length=config["training_config"]["context_length"], 
                                                                                                device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                                config=config,
                                                                                                mean=MEAN,
                                                                                                std=STD,
                                                                                                use_argmax=config["online_inference_config"]["use_argmax"],
                                                                                                create_video=False)

                                        returns.append(episode_return)
                                        t *= FRAME_SKIP
                                        ts.append(t)

                                        pbar.set_description(f"Online inference {color} {ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}")

                                    returns_mean = np.mean(returns)
                                    lifetime_mean = np.mean(ts)

                                    if wwandb:
                                        wandb.log({f"LifeTimeMean_{color}_{ret}": lifetime_mean, 
                                                f"ReturnsMean_{color}_{ret}": returns_mean})
                                    elif wcomet:
                                        experiment.log_metrics({f"LifeTimeMean_{color}_{ret}": lifetime_mean, 
                                                            f"ReturnsMean_{color}_{ret}": returns_mean}, step=it_counter)
        
                                return returns, ts

                            total_returns, total_ts = [], []
                            SKIP_RETURN = 4

                            # RED PILLAR
                            seeds_red = reds[::SKIP_RETURN]
                            red_returns, red_ts = optimize_pillar("red", seeds_red, config, wwandb, wcomet)
                            total_returns += red_returns
                            total_ts += red_ts

                            # GREEN PILLAR
                            seeds_green = greens[::SKIP_RETURN]
                            green_returns, green_ts = optimize_pillar("green", seeds_green, config, wwandb, wcomet)
                            total_returns += green_returns
                            total_ts += green_ts

                            total_returns = np.mean(total_returns)
                            total_ts = np.mean(total_ts)

                            if wwandb:
                                wandb.log({f"LifeTimeMean_{config['online_inference_config']['desired_return_1']}": total_ts, 
                                        f"ReturnsMean_{config['online_inference_config']['desired_return_1']}": total_returns})
                            elif wcomet:
                                experiment.log_metrics({f"LifeTimeMean_{config['online_inference_config']['desired_return_1']}": total_ts, 
                                                        f"ReturnsMean_{config['online_inference_config']['desired_return_1']}": total_returns}, step=it_counter)
                                
            model.train()
            wandb_step += 1 
            if wwandb:
                wandb.log({"checkpoint_step": epoch+1})
            elif wcomet:
                experiment.log_metrics({"checkpoint_step": epoch+1}, step=it_counter)
            torch.save(model.state_dict(), ckpt_path + str(epoch+1) + '_KTD.pth')

        elif config["model_config"]["mode"] == 'memory_maze':
            if tokens > tokens_cnt:
                tokens_cnt += 8_000_000
                if config["model_config"]["mode"] == 'memory_maze':
                    model.eval()
                    video_logged = False
                    with torch.no_grad():
                        seeds = np.arange(0, 100).tolist()[::20]
                        total_rew_mm = 0
                        cnt = 1
                        for ret in [config["online_inference_config"]["desired_return_1"]]:
                            attn_map_received = False
                            returns = []
                            ts = []
                            for i in range(len(seeds)):
                                episode_return, act_list, t, _, _, attn_map, frames = get_returns_MemoryMaze(model=model, ret=ret, seed=seeds[i], 
                                                                                        episode_timeout=episode_timeout, 
                                                                                        context_length=config["training_config"]["context_length"], 
                                                                                        device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                        config=config,
                                                                                        mean=None,
                                                                                        std=None,
                                                                                        use_argmax=config["online_inference_config"]["use_argmax"],
                                                                                        create_video=False)

                                returns.append(episode_return)
                                ts.append(t)
                                total_rew_mm += episode_return
                                curr_mean_ret = total_rew_mm / cnt
                                cnt += 1
                                pbar.set_description(f"Online inference_{ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")
                                # if wcomet:
                                #     if not video_logged:
                                #         pbar.set_description('Creating video...')
                                #         video_name = f'output_video_{config["text_description"]}.mp4'  # Number of frames in the video
                                #         frame_width = 64
                                #         frame_height = 64
                                #         frame_rate = 24  # Frames per second
                                #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                #         video = cv2.VideoWriter(video_name, fourcc, frame_rate, (frame_width, frame_height))

                                #         for frame in frames:
                                #             video.write(frame.astype(np.uint8)) 

                                #         video.release()

                                #         # experiment.log_asset(video_name)
                                #         experiment.log_video(video_name, overwrite=False)
                                #         video_logged = True

                            returns_mean = np.mean(returns)
                            returns_max = np.max(returns)
                            lifetime_mean = np.mean(ts)

                            if wwandb:
                                wandb.log({f"LifeTimeMean_{ret}":   lifetime_mean,
                                           f"ReturnsMax_{ret}":     returns_max,
                                           f"ReturnsMean_{ret}":    returns_mean})
                            elif wcomet:
                                experiment.log_metrics({f"LifeTimeMean_{ret}":   lifetime_mean,
                                           f"ReturnsMax_{ret}":     returns_max,
                                           f"ReturnsMean_{ret}":    returns_mean}, step=it_counter)
                                
                model.train()
                wandb_step += 1 
                if wwandb:
                    wandb.log({"checkpoint_step": tokens.item()})
                elif wcomet:
                    experiment.log_metrics({"checkpoint_step": tokens.item()}, step=it_counter)
                torch.save(model.state_dict(), ckpt_path + str(tokens.item()) + '_KTD.pth')

        elif config["model_config"]["mode"] == 'minigrid_memory' :
            if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or (epoch + 1) == 1:
                env_name = {'type': 'Minigrid', 'name': 'MiniGrid-MemoryS13Random-v0'}
                model.eval()
                video_logged = False
                with torch.no_grad():
                    for LENGTH in [31, 91]:
                        seeds = np.arange(0, 100).tolist()
                        total_rew_mm = 0
                        cnt = 1
                        for ret in [config["online_inference_config"]["desired_return_1"]]:
                            returns = []
                            ts = []
                            for i in range(len(seeds)):
                                episode_return, act_list, t, out_states, memories, attn_maps, attn_maps_seg, frames = get_returns_MinigridMemory(length=LENGTH, model=model, ret=ret, seed=seeds[i], 
                                                                                                                            episode_timeout=episode_timeout, 
                                                                                                                            context_length=config["training_config"]["context_length"], 
                                                                                                                            device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                                                            config=config,
                                                                                                                            mean=None,
                                                                                                                            std=None,
                                                                                                                            use_argmax=config["online_inference_config"]["use_argmax"],
                                                                                                                            create_video=False, env_name=env_name)

                                returns.append(episode_return)
                                ts.append(t)
                                total_rew_mm += episode_return
                                curr_mean_ret = total_rew_mm / cnt
                                cnt += 1
                                pbar.set_description(f"Online inference_r_{ret}_l_{LENGTH}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")

                            returns_mean = np.mean(returns)
                            returns_max = np.max(returns)
                            lifetime_mean = np.mean(ts)

                            if wwandb:
                                wandb.log({f"LifeTimeMean_r_{ret}_l_{LENGTH}":   lifetime_mean,
                                            f"ReturnsMax_r_{ret}_l_{LENGTH}":     returns_max,
                                            f"ReturnsMean_r_{ret}_l_{LENGTH}":    returns_mean})
                            elif wcomet:
                                experiment.log_metrics({f"LifeTimeMean_r_{ret}_l_{LENGTH}":   lifetime_mean,
                                            f"ReturnsMax_r_{ret}_l_{LENGTH}":     returns_max,
                                            f"ReturnsMean_r_{ret}_l_{LENGTH}":    returns_mean}, step=it_counter)
                                    
                    model.train()
                    wandb_step += 1 
                    if wwandb:
                        wandb.log({"checkpoint_step": epoch+1})
                    elif wcomet:
                        experiment.log_metrics({"checkpoint_step": epoch+1}, step=it_counter)
                    torch.save(model.state_dict(), ckpt_path + str(epoch+1) + '_KTD.pth')

            if epoch == config["training_config"]["epochs"] - 1:
                env_name = {'type': 'Minigrid', 'name': 'MiniGrid-MemoryS13Random-v0'}
                model.eval()
                with torch.no_grad():
                    for LENGTH in [11, 21, 31, 41, 51, 61, 71, 81, 91]:
                        seeds = np.arange(0, 100).tolist()
                        total_rew_mm = 0
                        cnt = 1
                        for ret in [config["online_inference_config"]["desired_return_1"]]:
                            returns = []
                            ts = []
                            for i in range(len(seeds)):
                                episode_return, act_list, t, out_states, memories, attn_maps, attn_maps_seg, frames = get_returns_MinigridMemory(length=LENGTH, model=model, ret=ret, seed=seeds[i], 
                                                                                                                            episode_timeout=episode_timeout, 
                                                                                                                            context_length=config["training_config"]["context_length"], 
                                                                                                                            device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                                                            config=config,
                                                                                                                            mean=None,
                                                                                                                            std=None,
                                                                                                                            use_argmax=config["online_inference_config"]["use_argmax"],
                                                                                                                            create_video=False, env_name=env_name)

                                returns.append(episode_return)
                                ts.append(t)
                                total_rew_mm += episode_return
                                curr_mean_ret = total_rew_mm / cnt
                                cnt += 1
                                pbar.set_description(f"Online inference_r_{ret}_l_{LENGTH}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")

                            returns_mean = np.mean(returns)
                            returns_max = np.max(returns)
                            lifetime_mean = np.mean(ts)

                            if wwandb:
                                wandb.log({f"FINAL_LifeTimeMean_r_{ret}_l_{LENGTH}":   lifetime_mean,
                                            f"FINAL_ReturnsMax_r_{ret}_l_{LENGTH}":     returns_max,
                                            f"FINAL_ReturnsMean_r_{ret}_l_{LENGTH}":    returns_mean})
                            elif wcomet:
                                experiment.log_metrics({f"FINAL_LifeTimeMean_r_{ret}_l_{LENGTH}":   lifetime_mean,
                                            f"FINAL_ReturnsMax_r_{ret}_l_{LENGTH}":     returns_max,
                                            f"FINAL_ReturnsMean_r_{ret}_l_{LENGTH}":    returns_mean}, step=it_counter)
        
        elif config["model_config"]["mode"] == 'maniskill-pushcube':
            if  ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or (epoch + 1) == 1:
            # if tokens > tokens_cnt:
            #     tokens_cnt += tokens_cnt_step
            #     if tokens_cnt > 50_000_000:
            #         tokens_cnt_step = 4_000_000
            #     elif tokens_cnt > 25_000_000:
            #         tokens_cnt_step = 4_000_000
                if config["model_config"]["mode"] == 'maniskill-pushcube' and config["training_config"]["online_inference"]:
                    model.eval()
                    logged_video_seeds = 0
                    bad_video_logged = False
                    medium_video_logged = False
                    good_video_logged = False
                    with torch.no_grad():
                        # seeds = np.arange(0, 100).tolist()[::5]
                        # seeds = [123, 231, 321, 777, 888]
                        seeds = [16, 41, 64, 73, 26, # hard seeds
                                 0, 93, 9, 15, 19] # easy seeds
                        total_rew_mm = 0
                        cnt = 1
                        for ret in [config["online_inference_config"]["desired_return_1"]]:
                            returns = []
                            ts = []

                            eval_metrics = defaultdict(list)
                            metrics_maniskill = {"success_once": [],
                                                 "return": [],
                                                 "episode_len": [],
                                                 "reward": [],
                                                 "success_at_end": []}

                            for i in range(len(seeds)):
                                try:
                                    episode_return, _, t, _, _, _, _, eval_metrics = get_returns_ManiSkill(env=env, model=model, ret=ret, seed=seeds[i], 
                                                                                            episode_timeout=episode_timeout, 
                                                                                            context_length=config["training_config"]["context_length"], 
                                                                                            device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                            config=config,
                                                                                            mean=None,
                                                                                            std=None,
                                                                                            use_argmax=config["online_inference_config"]["use_argmax"],
                                                                                            create_video=False,
                                                                                            sparse_reward=config["data_config"]["sparse_reward"])

                                    model.to(device)
                                    returns.append(episode_return)
                                    ts.append(t)

                                    # time.sleep(0.1)
                                    torch.cuda.empty_cache()
                                    

                                    for k, v in eval_metrics.items():
                                        if v:
                                            mean = torch.stack(v).float().mean().item()
                                            metrics_maniskill[k].append(mean)

                                    total_rew_mm += episode_return
                                    curr_mean_ret = total_rew_mm / max(cnt, 1)
                                    cnt += 1
                                    eval_output_dir = f"ManiSkill/val_videos/{config['model_mode']}/{config['text_description']}"
                                    video_files = glob.glob(os.path.join(eval_output_dir, "*.mp4"))
                                    if video_files:
                                        latest_video = max(video_files, key=os.path.getctime)

                                    if logged_video_seeds < 3:
                                        wandb.log({f"episode_video_{logged_video_seeds}": wandb.Video(latest_video)})
                                        logged_video_seeds += 1

                                    if not bad_video_logged and episode_return < 10:
                                        wandb.log({f"episode_video_bad": wandb.Video(latest_video)})
                                        bad_video_logged = True
                                    elif not medium_video_logged and 20 <= episode_return < 30:
                                        wandb.log({f"episode_video_medium": wandb.Video(latest_video)})
                                        medium_video_logged = True
                                    elif not good_video_logged and episode_return >= 40:
                                        wandb.log({f"episode_video_good": wandb.Video(latest_video)})
                                        good_video_logged = True

                                    pbar.set_description(f"Online inference_{ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")
                                except Exception as e:
                                    print(f"Error during evaluation of seed {seeds[i]}: {str(e)}")
                                    continue


                            for k, v in metrics_maniskill.items():
                                metrics_maniskill[k] = np.mean(v)
                            if wwandb:
                                for k, v in metrics_maniskill.items():
                                    wandb.log({f"eval/eval_{k}_mean_{ret}": v})
                            elif wcomet:
                                for k, v in metrics_maniskill.items():
                                    experiment.log_metrics({f"eval/eval_{k}_mean_{ret}": v}, step=it_counter)
                                
                model.train()
                wandb_step += 1 
                if wwandb:
                    wandb.log({"checkpoint_step": tokens.item()})
                elif wcomet:
                    experiment.log_metrics({"checkpoint_step": tokens.item()}, step=it_counter)
                torch.save(model.state_dict(), ckpt_path + str(tokens.item()) + '_KTD.pth')
    if env is not None:
        env.close()
    return model