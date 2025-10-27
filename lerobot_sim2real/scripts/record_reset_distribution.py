import json
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro

from lerobot_sim2real.envs.randomization_wrapper import (
    LightingRandomizationWrapper,
    DistractorObjectsWrapper,
)


@dataclass
class Args:
    env_id: str = "SO100GraspCube-v1"
    """The environment id to use"""
    record_dir: str = "videos"
    """Directory to save recordings of the camera captured images. If none no recordings are saved"""
    env_kwargs_json_path: Optional[str] = None
    """path to a json file containing additional environment kwargs"""
    num_resets: int = 100
    """Number of resets to record"""
def main(args: Args):
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="sensors",
        domain_randomization=True,
        reward_mode="none",
        # note this changes the sim camera resolution to 512x512, but this is just for development/debugging. During training
        # we will not use such large images
        sensor_configs=dict(width=512, height=512, shader_pack="default")
    )
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))

    # Filter out wrapper-specific kwargs that shouldn't be passed to gym.make()
    wrapper_kwargs = ['lighting_randomization', 'distractor_objects']
    gym_env_kwargs = {k: v for k, v in env_kwargs.items() if k not in wrapper_kwargs}

    env = gym.make(args.env_id, **gym_env_kwargs)

    # Apply domain randomization wrappers
    if env_kwargs.get('lighting_randomization', {}).get('enabled', False):
        env = LightingRandomizationWrapper(env, env_kwargs)

    if env_kwargs.get('distractor_objects', {}).get('enabled', False):
        env = DistractorObjectsWrapper(env, env_kwargs)

    if args.record_dir is not None:
        env = RecordEpisode(env, output_dir=args.record_dir, save_video=False, save_trajectory=False, video_fps=15)
    env.reset()
    for _ in range(args.num_resets):
        env.reset()
        env.render_images.append(env.capture_image())
    name = f"{args.env_id}_reset_distribution"
    env.flush_video(name=name)
    print(f"Saved video to {env.output_dir}/{name}.mp4")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)