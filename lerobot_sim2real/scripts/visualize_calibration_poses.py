"""
Visualize the calibration poses in simulation.
This helps you compare what the sim expects vs what your real robot looks like.
"""
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import tyro
import time
import json

import gymnasium as gym
import mani_skill.envs  # noqa
from mani_skill.utils import sapien_utils
from lerobot_sim2real.utils.camera_calibration import patch_camera_pose_from_quaternion
import matplotlib.pyplot as plt

from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent


@dataclass
class Args:
    env_id: str = "SO100GraspCube-v1"
    env_kwargs_json_path: str = "env_config.json"
    robot_id: str = "stone_home"


def main(args: Args):
    # The same poses used in so100.py calibration
    qpos_samples = [
        # Pose 1: Neutral upright
        np.array([0, 0, 0, np.pi / 2, np.pi / 2, 0.2]),
        # Pose 2: Rotated left, slightly lifted
        np.array([np.pi / 3, -np.pi / 6, 0, np.pi / 2, np.pi / 2, 0]),
        # Pose 3: Rotated right
        np.array([-np.pi / 3, 0, 0, np.pi / 2, np.pi / 2, 0.2]),
        # Pose 4: Arm extended forward and down
        np.array([0, np.pi / 4, -np.pi / 4, np.pi / 3, np.pi / 2, 0]),
        # Pose 5: Arm tucked, rotated left
        np.array([np.pi / 4, -np.pi / 4, np.pi / 4, np.pi / 2, 0, 0.2]),
        # Pose 6: Arm extended right side
        np.array([-np.pi / 4, np.pi / 6, -np.pi / 6, np.pi / 2, np.pi, 0]),
        # Pose 7: Different wrist angle
        np.array([np.pi / 6, 0, 0, np.pi / 4, np.pi / 2, 0.2]),
        # Pose 8: Arm stretched out
        np.array([0, np.pi / 3, -np.pi / 3, np.pi / 4, np.pi / 2, 0]),
    ]
    
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    print("=" * 60)
    print("CALIBRATION POSE VIEWER")
    print("=" * 60)
    print("\nThis shows you the expected joint positions in RADIANS and DEGREES")
    print("Compare these to what your real robot shows in LeRobot.\n")
    
    for i, qpos in enumerate(qpos_samples):
        print(f"\n--- Pose {i+1} ---")
        for j, (name, val) in enumerate(zip(joint_names, qpos)):
            print(f"  {name:15s}: {val:7.3f} rad = {np.degrees(val):7.1f} deg")
    
    print("\n" + "=" * 60)
    print("Now launching simulation to visualize each pose...")
    print("=" * 60)
    
    # Load env config (only use supported keys)
    with open(args.env_kwargs_json_path, 'r') as f:
        full_env_kwargs = json.load(f)
    
    # Filter to only supported kwargs
    supported_keys = ["base_camera_settings", "greenscreen_overlay_path", "spawn_box_pos", 
                      "spawn_box_half_size", "domain_randomization_config"]
    env_kwargs = {k: v for k, v in full_env_kwargs.items() if k in supported_keys}
    
    # Create sim environment
    sim_env = gym.make(
        args.env_id,
        obs_mode="rgb",
        render_mode="sensors",
        num_envs=1,
        **env_kwargs
    )
    patch_camera_pose_from_quaternion(sim_env)  # Apply quaternion from config if present
    sim_env.reset()
    
    # Connect to real robot using proper wrapper
    print("\nConnecting to real robot...")
    real_robot = create_real_robot(uid="so100")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)
    
    # Create wrapped environment for proper observation handling
    sim_env_wrapped = FlattenRGBDObservationWrapper(sim_env)
    real_env = Sim2RealEnv(
        sim_env=sim_env_wrapped, 
        agent=real_agent, 
        skip_data_checks=True,
        real_reset_function=lambda self, seed, options: None  # Don't move robot on reset
    )
    
    # Set a good camera view for visualization
    cam_pos = np.array([0.5, 0.3, 0.5])
    cam_target = np.array([0.0, 0.0, 0.15])
    pose = sapien_utils.look_at(cam_pos, cam_target)
    sim_env.unwrapped.camera_mount.set_pose(pose)
    
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, qpos in enumerate(qpos_samples):
        print(f"\n\n{'='*60}")
        print(f"POSE {i+1}/{len(qpos_samples)}")
        print(f"{'='*60}")
        
        # Print expected joint values
        print("\nExpected joint positions (what sim will show):")
        for j, (name, val) in enumerate(zip(joint_names, qpos)):
            print(f"  {name:15s}: {val:7.3f} rad = {np.degrees(val):7.1f} deg")
        
        # Set sim robot to this pose
        articulation = sim_env.unwrapped.agent.robot
        current_qpos = articulation.get_qpos()
        current_qpos[0, :6] = qpos  # Set the arm joints
        articulation.set_qpos(current_qpos)
        
        # Move real robot to match
        print("\nMoving REAL robot to this pose...")
        action = np.array(qpos, dtype=np.float32)
        
        # Move robot slowly - send action multiple times
        steps = 30
        for step in range(steps):
            real_agent.send_action(action)
            time.sleep(0.05)
        
        time.sleep(0.5)  # Let robot settle
        
        # Read what the real robot reports
        real_state = real_agent.get_state()
        print("\nReal robot reports:")
        for j, name in enumerate(joint_names):
            if j < len(real_state):
                real_val = real_state[j]
                expected_val = qpos[j]
                diff = real_val - expected_val
                print(f"  {name:15s}: {real_val:7.3f} rad = {np.degrees(real_val):7.1f} deg  (diff: {np.degrees(diff):+.1f} deg)")
        
        # Render sim
        sim_env.unwrapped.scene.update_render()
        sim_env.unwrapped._sensors["base_camera"].camera.take_picture()
        sim_img = sim_env.unwrapped._sensors["base_camera"].camera.get_picture("Color")[..., :3]
        sim_img = np.clip(sim_img * 255, 0, 255).astype(np.uint8)
        
        # Get real camera image through the real_env
        real_obs = real_env.get_obs()
        if "rgb" in real_obs:
            real_img = real_obs["rgb"].cpu().numpy()[0].astype(np.uint8)
        elif "sensor_data" in real_obs:
            # Try to get from sensor_data
            sensor_data = real_obs["sensor_data"]
            for cam_name in sensor_data:
                if "rgb" in sensor_data[cam_name]:
                    real_img = sensor_data[cam_name]["rgb"][0].cpu().numpy().astype(np.uint8)
                    break
        else:
            print("Warning: Could not find RGB image in observation")
            real_img = np.zeros_like(sim_img)
        
        # Display
        axes[0].clear()
        axes[0].imshow(sim_img)
        axes[0].set_title(f"SIM - Pose {i+1}")
        axes[0].axis('off')
        
        axes[1].clear()
        axes[1].imshow(real_img)
        axes[1].set_title(f"REAL - Pose {i+1}")
        axes[1].axis('off')
        
        fig.suptitle(f"Pose {i+1}/{len(qpos_samples)} - Compare joint positions!\nPress ENTER in terminal to continue...")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        input(f"\nPress ENTER to continue to pose {i+2}...")
    
    print("\n" + "=" * 60)
    print("DONE! Based on the differences, you need to set CALIBRATION_OFFSET")
    print("in so100.py to: offset = real_reported - expected (in degrees)")
    print("=" * 60)
    
    real_robot.disconnect()
    real_env.close()
    sim_env.close()
    plt.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
