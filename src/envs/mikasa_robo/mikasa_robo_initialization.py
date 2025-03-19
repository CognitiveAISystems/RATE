import gymnasium as gym
import mikasa_robo_suite
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info
from mikasa_robo_suite.dataset_collectors.get_dataset_collectors_ckpt import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

class InitializeMikasaRoboEnv:
    @staticmethod
    def create_mikasa_robo_env(env_name, run_dir, config):

        # Extract actual env name from full name
        env_name = env_name.split("_")[-1]
        
        # Create base environment
        # env = gym.make(env_name, num_envs=1, obs_mode="rgb", render_mode="all", sim_backend="gpu")
        env = gym.make(env_name, num_envs=16, obs_mode="rgb", render_mode="all", sim_backend="gpu")

        # Apply state wrappers
        state_wrappers_list, episode_timeout = env_info(env_name)
        print(f"Episode timeout: {episode_timeout}")
        for wrapper_class, wrapper_kwargs in state_wrappers_list:
            env = wrapper_class(env, **wrapper_kwargs)

        # Apply observation wrapper
        env = FlattenRGBDObservationWrapper(
            env, rgb=True, depth=False, state=False, 
            oracle=False, joints=False
        )

        # Flatten action space if needed
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)

        # Add recording capability
        env = RecordEpisode(
            env, 
            output_dir=f"{run_dir}/videos", 
            save_trajectory=False, 
            trajectory_name="",
            max_steps_per_video=config["online_inference"]["episode_timeout"], 
            video_fps=30
        )

        # Wrap in vector env
        env = ManiSkillVectorEnv(env, 16, ignore_terminations=True, record_metrics=True)

        return env