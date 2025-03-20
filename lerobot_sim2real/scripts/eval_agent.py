from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
from lerobot.common.robot_devices.cameras.configs import (
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
import torch
import tyro
from lerobot_sim2real.rl.agents.actor_critic import ActorCritic
from lerobot_sim2real.rl.buffer.dict_array import DictArray
from lerobot_sim2real.models.vision import NatureCNN

import mani_skill.envs.tasks.digital_twins.koch_arm.grasp_cube
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm

@dataclass
class Args:
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to load agent weights from for evaluation. If None then a random agent will be used"""

def main(args: Args):
    robot_config = KochRobotConfig(
        leader_arms={},
        follower_arms={
            "main": DynamixelMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        },
        cameras={
            "base_camera": IntelRealSenseCameraConfig(
                serial_number=146322070293,
                fps=30,
                width=640,
                height=480,
            ),
        },
        calibration_dir="calibration/koch",
    )
    real_robot = ManipulatorRobot(robot_config)

    # max control freq for lerobot really is just 60Hz
    real_agent = LeRobotRealAgent(real_robot)


    max_episode_steps = 200
    sim_env = gym.make(
        "KochGraspCube-v1",
        obs_mode="rgb+segmentation",
        sim_config={"sim_freq": 120, "control_freq": 15},
        render_mode="sensors", # only sensors mode is supported right now for real envs, basically rendering the direct visual observations fed to policy
        max_episode_steps=max_episode_steps, # give our robot more time to try and re-try the task
    )
    # you can apply most wrappers freely to the sim_env and the real env will use them
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    sim_env = RecordEpisode(sim_env, output_dir="videos", save_trajectory=False, video_fps=sim_env.unwrapped.control_freq)
    
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, obs_mode="rgb")
    sim_env.print_sim_details()
    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()

    for k in sim_obs.keys():
        print(
            f"{k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}"
        )

    # load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ActorCritic(sim_env, sample_obs=real_obs, feature_net=NatureCNN(real_obs))
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded agent from {args.checkpoint}")
    else:
        print("No checkpoint provided, using random agent")

    pbar = tqdm(range(max_episode_steps))
    done = False
    while not done:
        real_obs, _, terminated, truncated, info = real_env.step(agent.get_action(real_obs))
        done = terminated or truncated
        pbar.update(1)
    sim_env.close()
    real_agent.stop()

    print("Saved video to videos/0.mp4")

if __name__ == "__main__":
    tyro.cli(main)