import torch
import numpy as np
from collections import defaultdict
import glob
import os

from .base_trainer import BaseTrainer


reds = [
    2, 3, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20, 21, 25, 26, 27, 28, 29, 31, 38, 40, 41, 42, 45,
    46, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 61, 63, 64, 67, 68, 70, 72, 73, 74, 77, 80, 82, 84, 
    86, 88, 89, 90, 91, 92, 97, 98, 99, 100, 101, 103, 106, 108, 109, 113, 115, 116, 117, 120, 123, 
    124, 125, 126, 127, 128, 129, 133, 134, 136, 139, 140, 142, 144, 145, 147, 148, 151, 152, 153, 
    154, 156, 157, 158, 159, 161, 164, 165, 170, 171, 173
]

greens = [
    0, 1, 4, 5, 7, 12, 13, 19, 22, 23, 24, 30, 32, 33, 34, 35, 36, 37, 39, 43, 44, 47, 48, 56, 57,
    62, 65, 66, 69, 71, 75, 76, 78, 79, 81, 83, 85, 87, 93, 94, 95, 96, 102, 104, 105, 107, 110, 111, 
    112, 114, 118, 119, 121, 122, 130, 131, 132, 135, 137, 138, 141, 143, 146, 149, 150, 155, 160, 162, 
    163, 166, 167, 168, 169, 172, 175, 176, 177, 182, 183, 187, 190, 192, 193, 195, 199, 204, 206, 208, 
    209, 210, 212, 215, 216, 218, 219, 220, 221, 223, 224, 225
]


class InferenceHandler(BaseTrainer):
    @staticmethod
    def perform_mini_inference_tmaze(self, episode_timeout, corridor_length, text=None, env=None):
        from validation.val_tmaze import get_returns_TMaze
        from utils.set_seed import seeds_list

        self.model.eval()
        with torch.no_grad():
            batch_size = len(seeds_list)
            if "multiple_timeouts" not in self.config["online_inference"]:
                # Define the list of multipliers to test
                multipliers = [9, 30, 90, 150, 300, 600, 900, 1200, 2400, 4800, 9600]
                
                for multiplier in multipliers:
                    # inference on corridor_length = train corridor_length * multiplier
                    rewards, successes = get_returns_TMaze(
                        model=self.model,
                        ret=1.0,
                        seeds=seeds_list,
                        episode_timeout=1*multiplier,
                        corridor_length=1*multiplier-2,
                        context_length=self.config["training"]["context_length"],
                        device=self.device,
                        config=self.config,
                        create_video=False,
                    )

                    episode_return = sum(rewards)/batch_size

                    if self.wwandb:
                        if text is None:
                            self.log({
                                f"Success_rate_T_{multiplier}": episode_return,
                            })
                        else:
                            self.log({
                                f"Success_rate_S_{text}_x{multiplier}": episode_return,
                            })
                    
                    if self.config["model_mode"] in ["RATE", "MATL"]:
                        self.current_metric_value = episode_return
                    else:
                        # self.current_metric_value = episode_return_1x
                        self.current_metric_value = episode_return
                    
                    print(f"----- [T: {1*multiplier}] | Success rate: {episode_return} | \n")

    @staticmethod
    def perform_mini_inference_vizdoom(self, episode_timeout, text=None, env=None):
        from src.validation.val_vizdoom_two_colors import get_returns_VizDoom

        self.model.eval()
        with torch.no_grad():
            FRAME_SKIP = 2
            def optimize_pillar(color, seeds, config):
                for ret in [config["online_inference"]["desired_return_1"]]:
                    returns = []
                    ts = []
                    for i in range(len(seeds)):
                        episode_return, t = \
                            get_returns_VizDoom(
                                model=self.model, 
                                ret=ret,
                                seed=seeds[i], 
                                episode_timeout=episode_timeout, 
                                context_length=self.config["training"]["context_length"], 
                                device=self.device, 
                                config=self.config,
                                use_argmax=self.config["online_inference"]["use_argmax"],
                                create_video=False,
                            )

                        returns.append(episode_return)
                        t *= FRAME_SKIP
                        ts.append(t)

                        self.pbar.set_description(f"Online inference {color} {ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}")

                    returns_mean = np.mean(returns)
                    lifetime_mean = np.mean(ts)

                    if self.wwandb:
                        self.log({
                            f"LifeTimeMean_{color}_{ret}": lifetime_mean, 
                            f"ReturnsMean_{color}_{ret}": returns_mean
                        })

                return returns, ts

            total_returns, total_ts = [], []
            SKIP_RETURN = 4

            # RED PILLAR
            seeds_red = reds[::SKIP_RETURN]
            red_returns, red_ts = optimize_pillar("red", seeds_red, self.config)
            total_returns += red_returns
            total_ts += red_ts

            # GREEN PILLAR
            seeds_green = greens[::SKIP_RETURN]
            green_returns, green_ts = optimize_pillar("green", seeds_green, self.config)
            total_returns += green_returns
            total_ts += green_ts

            total_returns = np.mean(total_returns)
            total_ts = np.mean(total_ts)

            if self.wwandb:
                self.log({
                    f"LifeTimeMean_{self.config['online_inference']['desired_return_1']}": total_ts, 
                    f"ReturnsMean_{self.config['online_inference']['desired_return_1']}": total_returns
                })
            self.current_metric_value = total_returns

    @staticmethod
    def perform_mini_inference_minigridmemory(self, episode_timeout, text=None, env=None):
        from validation.val_minigridmemory import get_returns_MinigridMemory

        self.model.eval()
        with torch.no_grad():
            if "Random_True" in self.config["data"]["path_to_dataset"]:
                env_name = {'type': 'Minigrid', 'name': 'MiniGrid-MemoryS13Random-v0'}
            else:
                env_name = {'type': 'Minigrid', 'name': 'MiniGrid-MemoryS13-v0'}
            returns_mean_41, returns_mean_91 = 0, 0
            for LENGTH in [41, 91]:
                seeds = np.arange(0, 100).tolist()
                total_rew_mm = 0
                cnt = 1
                for ret in [self.config["online_inference"]["desired_return_1"]]:
                    returns = []
                    ts = []
                    for i in range(len(seeds)):
                        episode_return, t = \
                            get_returns_MinigridMemory(
                                length=LENGTH, 
                                model=self.model, 
                                ret=ret, 
                                seed=seeds[i], 
                                episode_timeout=episode_timeout, 
                                context_length=self.config["training"]["context_length"], 
                                device=self.device, 
                                config=self.config,
                                use_argmax=self.config["online_inference"]["use_argmax"],
                                create_video=False,
                                env_name=env_name,
                            )

                        returns.append(episode_return)
                        ts.append(t)
                        total_rew_mm += episode_return
                        curr_mean_ret = total_rew_mm / cnt
                        cnt += 1
                        self.pbar.set_description(f"Online inference_r_{ret}_l_{LENGTH}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")

                    returns_mean = np.mean(returns)
                    returns_max = np.max(returns)
                    lifetime_mean = np.mean(ts)

                    if self.wwandb:
                        self.log({
                            f"LifeTimeMean_r_{ret}_l_{LENGTH}": lifetime_mean,
                            f"ReturnsMax_r_{ret}_l_{LENGTH}": returns_max,
                            f"ReturnsMean_r_{ret}_l_{LENGTH}": returns_mean
                        })
                if LENGTH == 41:
                    returns_mean_41 = returns_mean
                if LENGTH == 91:
                    returns_mean_91 = returns_mean
            if self.config["model_mode"] in ["RATE"]:
                self.current_metric_value = returns_mean_41
            else:
                self.current_metric_value = returns_mean_91
                        
    @staticmethod
    def perform_mini_inference_memorymaze(self, episode_timeout, text=None, env=None):
        from src.validation.val_memory_maze import get_returns_MemoryMaze

        self.model.eval()
        with torch.no_grad():
            SKIP_RETURN = 20
            seeds = np.arange(0, 100).tolist()[::SKIP_RETURN]
            total_rew_mm = 0
            cnt = 1
            for ret in [self.config["online_inference"]["desired_return_1"]]:
                returns = []
                ts = []
                for i in range(len(seeds)):
                    episode_return, act_list, t, _, _, attn_map, frames = \
                        get_returns_MemoryMaze(
                            model=self.model, 
                            ret=ret, seed=seeds[i],
                            episode_timeout=episode_timeout,
                            context_length=self.config["training"]["context_length"], 
                            device=self.device,
                            config=self.config,
                            use_argmax=self.config["online_inference"]["use_argmax"],
                            create_video=False
                        )

                    returns.append(episode_return)
                    ts.append(t)
                    total_rew_mm += episode_return
                    curr_mean_ret = total_rew_mm / cnt
                    cnt += 1
                    self.pbar.set_description(f"Online inference_{ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")

                returns_mean = np.mean(returns)
                returns_max = np.max(returns)
                lifetime_mean = np.mean(ts)

                if self.wwandb:
                    self.log({
                        f"LifeTimeMean_{ret}": lifetime_mean,
                        f"ReturnsMax_{ret}": returns_max,
                        f"ReturnsMean_{ret}": returns_mean})
                self.current_metric_value = returns_mean
                    
    @staticmethod
    def perform_mini_inference_popgym(self, episode_timeout, text=None, env=None):
        from src.validation.val_popgym import get_returns_POPGym

        self.model.eval()
        with torch.no_grad():
            SKIP_RETURN = 1
            seeds = np.arange(0, 100).tolist()[::SKIP_RETURN]
            total_rew_mm = 0
            cnt = 1
            for ret in [self.config["online_inference"]["desired_return_1"]]:
                returns = []
                ts = []
                for i in range(len(seeds)):
                    episode_return, act_list, t, _, _, attn_map, frames = \
                        get_returns_POPGym(
                            model=self.model, 
                            ret=ret, seed=seeds[i],
                            episode_timeout=episode_timeout,
                            context_length=self.config["training"]["context_length"], 
                            device=self.device,
                            config=self.config,
                            use_argmax=self.config["online_inference"]["use_argmax"]
                        )

                    returns.append(episode_return)
                    ts.append(t)
                    total_rew_mm += episode_return
                    curr_mean_ret = total_rew_mm / cnt
                    cnt += 1
                    self.pbar.set_description(f"Online inference_{ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")

                returns_mean = np.mean(returns)
                returns_max = np.max(returns)
                lifetime_mean = np.mean(ts)

                if self.wwandb:
                    self.log({
                        f"LifeTimeMean_{ret}": lifetime_mean,
                        f"ReturnsMax_{ret}": returns_max,
                        f"ReturnsMean_{ret}": returns_mean})
                self.current_metric_value = returns_mean

        self.model.to(self.device)

    @staticmethod
    def perform_mini_inference_arshot(self, episode_timeout, text=None, env=None):
        from src.validation.val_arshot import get_returns_ARShot

        self.model.eval()
        with torch.no_grad():
            SKIP_RETURN = 10
            seeds = np.arange(0, 100).tolist()[::SKIP_RETURN]
            
            # Test different n_pairs configurations
            # n_pairs_configs = [6, 10, 15, 20]
            # shot_modes = ["after_pairs", "after_any_colon"]
            n_pairs_configs = [2, 20, 50, 100, 250]
            shot_modes = ["after_pairs"]
            
            for n_pairs in n_pairs_configs:
                for shot_mode in shot_modes:
                    for ret in [self.config["online_inference"]["desired_return_1"]]:
                        # Pass all seeds at once for batch processing
                        episode_returns, successes = get_returns_ARShot(
                            model=self.model, 
                            ret=ret, 
                            seeds=seeds,  # Pass all seeds at once
                            n_pairs=n_pairs,
                            shot_mode=shot_mode,
                            episode_timeout=5*n_pairs+4+1,
                            context_length=self.config["training"]["context_length"], 
                            device=self.device,
                            config=self.config,
                            use_argmax=self.config["online_inference"]["use_argmax"],
                            deterministic_vocab=True,
                            full_universe_vocab=True,
                            randomize_pairs=True,
                            include_pass_token=True,
                            max_vocab_size=self.config["max_vocab_size"]
                        )

                        returns_mean = np.mean(episode_returns)
                        success_rate = np.mean([1.0 if r == 1.0 else 0.0 for r in episode_returns])

                        if self.wwandb:
                            self.log({
                                f"ARShot_n_pairs_{n_pairs}_{shot_mode}_SR": success_rate,
                            })
                        
                        print(f"n_pairs={n_pairs}, mode={shot_mode}: Success rate = {success_rate:.3f}, Mean return = {returns_mean:.3f}")
                        
                        # Use the most challenging configuration for best checkpoint selection
                        self.current_metric_value = success_rate

        self.model.to(self.device)

    @staticmethod
    def perform_mini_inference_mikasarobo(self, episode_timeout, text=None, env=None):
        from src.validation.val_mikasa_robo import get_returns_MIKASARobo

        self.model.eval()
        with torch.no_grad():
            SKIP_RETURN = 10
            # seeds = np.arange(0, 100).tolist()[::SKIP_RETURN]
            # seeds = [2, 3, 1, 9, 0, 4]
            total_rew_mm = 0
            cnt = 1
            for ret in [self.config["online_inference"]["desired_return_1"]]:
                returns = []
                ts = []

                eval_metrics = defaultdict(list)
                metrics_maniskill = {
                    "success_once": [],
                    "return": [],
                    "episode_len": [],
                    "reward": [],
                    "success_at_end": []
                }

                # In ManiSkill we can process parallel episodes, and therfore
                # we can proceess all eval epiisodes at once
                for i in [42]:
                    episode_return, _, t, _, _, _, _, eval_metrics = \
                        get_returns_MIKASARobo(
                            env=env, 
                            model=self.model, 
                            ret=ret, seed=i, 
                            episode_timeout=episode_timeout, 
                            context_length=self.config["training"]["context_length"], 
                            device=self.device, 
                            config=self.config,
                            use_argmax=self.config["online_inference"]["use_argmax"],
                            create_video=False
                        )
                    env.close()

                    ts.append(t)
                    
                    for k, v in eval_metrics.items():
                        if v:
                            mean = torch.stack(v).float().mean().item()
                            metrics_maniskill[k].append(mean)

                    total_rew_mm += episode_return
                    curr_mean_ret = total_rew_mm / max(cnt, 1)
                    cnt += 1

                    self.pbar.set_description(f"Online inference_{ret}: [-] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")

                for k, v in metrics_maniskill.items():
                    metrics_maniskill[k] = np.mean(v)
                if self.wwandb:
                    print(f"Metrics: {metrics_maniskill}")
                    for k, v in metrics_maniskill.items():
                        self.log({f"eval/eval_{k}_mean": v})
                    self.log({f"eval/return_to_go": ret})
                    self.log({"success_once": metrics_maniskill['success_once']})
                self.current_metric_value = metrics_maniskill['success_once']
        
        torch.cuda.empty_cache()
        self.model.to(self.device)

    @staticmethod
    def perform_mini_inference_mdp(self, episode_timeout, text=None, env=None):
        from src.validation.val_mdp import get_returns_MDP

        self.model.eval()
        with torch.no_grad():
            SKIP_RETURN = 1
            seeds = np.arange(0, 100).tolist()[::SKIP_RETURN]
            total_rew_mm = 0
            cnt = 1
            for ret in [self.config["online_inference"]["desired_return_1"]]:
                returns = []
                ts = []
                for i in range(len(seeds)):
                    episode_return, act_list, t, _, _, attn_map, frames = \
                        get_returns_MDP(
                            model=self.model, 
                            ret=ret, seed=seeds[i],
                            episode_timeout=episode_timeout,
                            context_length=self.config["training"]["context_length"], 
                            device=self.device,
                            config=self.config,
                            use_argmax=self.config["online_inference"]["use_argmax"],
                            env_name=self.config["model"]["env_name"]
                        )

                    returns.append(episode_return)
                    ts.append(t)
                    total_rew_mm += episode_return
                    curr_mean_ret = total_rew_mm / cnt
                    cnt += 1
                    self.pbar.set_description(f"Online inference_{ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}, Total Return: {total_rew_mm:.2f}, Current Mean Return: {curr_mean_ret:.2f}")

                returns_mean = np.mean(returns)
                returns_max = np.max(returns)
                lifetime_mean = np.mean(ts)

                if self.wwandb:
                    self.log({
                        f"LifeTimeMean_{ret}": lifetime_mean,
                        f"ReturnsMax_{ret}": returns_max,
                        f"ReturnsMean_{ret}": returns_mean})
                self.current_metric_value = returns_mean

        self.model.to(self.device)

