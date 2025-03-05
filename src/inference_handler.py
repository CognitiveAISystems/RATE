import torch
import numpy as np

from .base_trainer import BaseTrainer


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


class InferenceHandler(BaseTrainer):
    @staticmethod
    def perform_mini_inference_tmaze(self, episode_timeout, corridor_length, text=None):
        from validation.val_tmaze import get_returns_TMaze
        from utils.set_seed import seeds_list

        self.model.eval()
        with torch.no_grad():
            desired_reward = 1.0
            goods, bads = 0, 0
            timers = []
            rewards = []
            pbar2 = range(len(seeds_list))

            for indx, iii in enumerate(pbar2):
                episode_return, act_list, t, _, delta_t, attn_map = \
                    get_returns_TMaze(
                        model=self.model,
                        ret=desired_reward,
                        seed=seeds_list[iii],
                        episode_timeout=episode_timeout,
                        corridor_length=corridor_length,
                        context_length=self.config["training"]["context_length"],
                        device=self.device,
                        config=self.config,
                        create_video=False,
                    )

                if episode_return == desired_reward:
                    goods += 1
                else:
                    bads += 1

                timers.append(delta_t)
                rewards.append(episode_return)

                self.pbar.set_description(f"[{text}] | [inference | {indx+1}/{len(seeds_list)}] ep {self.epoch+1} reward {episode_return}")

            suc_rate = goods / (goods + bads)
            ep_time = np.mean(timers)

            if self.wwandb:
                if text is None:
                    self.log({
                        "Success_rate": suc_rate,
                        "Mean_D[time]": ep_time
                    })
                else:
                    self.log({
                        f"Success_rate_S_{text}": suc_rate,
                        f"Mean_D[time]_S_{text}": ep_time
                    })

    @staticmethod
    def perform_mini_inference_vizdoom(self, episode_timeout, text=None):
        from src.validation.val_vizdoom_two_colors import get_returns_VizDoom

        self.model.eval()
        with torch.no_grad():
            FRAME_SKIP = 2
            def optimize_pillar(color, seeds, config):
                for ret in [config["online_inference"]["desired_return_1"]]:
                    returns = []
                    ts = []
                    for i in range(len(seeds)):
                        episode_return, act_list, t, _, _, attn_map = \
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

    @staticmethod
    def perform_mini_inference_minigridmemory(self, episode_timeout, text=None):
        from validation.val_minigridmemory import get_returns_MinigridMemory

        self.model.eval()
        with torch.no_grad():
            env_name = {'type': 'Minigrid', 'name': 'MiniGrid-MemoryS13Random-v0'}
            for LENGTH in [31, 91]:
                seeds = np.arange(0, 100).tolist()
                total_rew_mm = 0
                cnt = 1
                for ret in [self.config["online_inference"]["desired_return_1"]]:
                    returns = []
                    ts = []
                    for i in range(len(seeds)):
                        episode_return, act_list, t, out_states, memories, attn_maps, attn_maps_seg, frames = \
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
                            f"ReturnsMean_r_{ret}_l_{LENGTH}": returns_mean})
                        
    @staticmethod
    def perform_mini_inference_memorymaze(self, episode_timeout, text=None):
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