import json
import os
import time
from pathlib import Path
from typing import Optional
import gymnasium as gym
import sapien
from lerobot_sim2real.utils.safety import setup_safe_exit
from lerobot_sim2real.utils.camera_calibration import load_calibrated_pose, load_intrinsics, apply_intrinsics_to_camera
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
import cv2
import numpy as np
import tyro
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils import sapien_utils
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Default calibration path relative to the project root
DEFAULT_CALIBRATION_DIR = Path(__file__).parent.parent.parent / "simple-easyhec" / "results"

@dataclass
class Args:
    env_id: str = "SO100GraspCube-v1"
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    extrinsics_path: Optional[str] = None
    """Path to camera_extrinsic_ros.npy from easyhec calibration. If provided, auto-sets camera pose."""
    intrinsics_path: Optional[str] = None
    """Path to camera_intrinsic.json from calibration. If provided, sets exact camera intrinsics (fx, fy, cx, cy)."""
    use_existing_calibration: bool = True
    """If True, will check for existing calibration files and offer to use them. Set to False to skip this check."""
    robot_id: str = "so100"
    """Robot ID to look for calibration files."""


def find_existing_calibrations(robot_id: str) -> list[tuple[str, Path]]:
    """
    Find all existing calibration files for the given robot.
    
    Returns:
        List of (calibration_name, extrinsics_path) tuples
    """
    robot_dir = DEFAULT_CALIBRATION_DIR / robot_id
    if not robot_dir.exists():
        return []
    
    calibrations = []
    for subdir in robot_dir.iterdir():
        if subdir.is_dir():
            extrinsics_path = subdir / "base_camera" / "camera_extrinsic_ros.npy"
            if extrinsics_path.exists():
                calibrations.append((subdir.name, extrinsics_path))
    
    return sorted(calibrations, key=lambda x: x[0])


def prompt_for_calibration(calibrations: list[tuple[str, Path]]) -> Optional[Path]:
    """
    Prompt the user to select from available calibrations.
    
    Returns:
        Selected extrinsics path, or None if user wants manual adjustment
    """
    print("\n=== Existing Camera Calibrations Found ===")
    for i, (name, path) in enumerate(calibrations, 1):
        print(f"  [{i}] {name}")
    print(f"  [0] Skip - use manual adjustment only")
    print()
    
    while True:
        try:
            choice = input("Select calibration to use (or 0 to skip): ").strip()
            if choice == "0" or choice == "":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(calibrations):
                selected_name, selected_path = calibrations[idx]
                print(f"Using calibration: {selected_name}")
                return selected_path
            print(f"Invalid choice. Enter 0-{len(calibrations)}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            return None

def overlay_envs(sim_env, real_env):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
    """
    real_obs = real_env.get_obs()["sensor_data"]
    sim_obs = sim_env.get_obs()["sensor_data"]
    assert sorted(real_obs.keys()) == sorted(
        sim_obs.keys()
    ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

    overlaid_dict = sim_env.get_obs()["sensor_data"]
    overlaid_imgs = []
    for name in overlaid_dict:
        real_imgs = real_obs[name]["rgb"][0] / 255
        sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)

    return tile_images(overlaid_imgs)


def update_camera(sim_env):
    global camera_offset, rotation_offset, fov_offset, last_frame_time, help_message_printed, calibrated_base_pose
    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    # Reset camera position, rotation, and FOV on backspace
    if "backspace" in active_keys:
        camera_offset = np.zeros(3, dtype=np.float32)
        rotation_offset = np.zeros(3, dtype=np.float32)
        fov_offset = 0.0

    # Camera movement mapping based on active keys
    if "s" in active_keys:
        camera_offset[0] -= MOVEMENT_SPEED * delta_time  # Move forward
    if "w" in active_keys:
        camera_offset[0] += MOVEMENT_SPEED * delta_time  # Move back
    if "a" in active_keys:
        camera_offset[1] += MOVEMENT_SPEED * delta_time  # Move right
    if "d" in active_keys:
        camera_offset[1] -= MOVEMENT_SPEED * delta_time  # Move left
    if "up" in active_keys:
        camera_offset[2] += MOVEMENT_SPEED * delta_time  # Move up
    if "down" in active_keys:
        camera_offset[2] -= MOVEMENT_SPEED * delta_time  # Move down

    # Rotation controls: Q/E for yaw, R/F for pitch, Z/X for roll
    if "q" in active_keys:
        rotation_offset[2] += ROTATION_SPEED * delta_time  # Yaw left
    if "e" in active_keys:
        rotation_offset[2] -= ROTATION_SPEED * delta_time  # Yaw right
    if "r" in active_keys:
        rotation_offset[1] += ROTATION_SPEED * delta_time  # Pitch up
    if "f" in active_keys:
        rotation_offset[1] -= ROTATION_SPEED * delta_time  # Pitch down
    if "z" in active_keys:
        rotation_offset[0] += ROTATION_SPEED * delta_time  # Roll left
    if "x" in active_keys:
        rotation_offset[0] -= ROTATION_SPEED * delta_time  # Roll right

    # FOV control
    if "left" in active_keys:
        fov_offset -= FOV_CHANGE_SPEED * delta_time
    if "right" in active_keys:
        fov_offset += FOV_CHANGE_SPEED * delta_time

    # update camera position and fov
    if calibrated_base_pose is not None:
        # Use calibrated pose as base
        # Convert tensors to numpy arrays for sapien.Pose
        base_p = calibrated_base_pose.p.cpu().numpy().flatten() if hasattr(calibrated_base_pose.p, 'cpu') else np.array(calibrated_base_pose.p).flatten()
        base_q = calibrated_base_pose.q.cpu().numpy().flatten() if hasattr(calibrated_base_pose.q, 'cpu') else np.array(calibrated_base_pose.q).flatten()
        
        # Apply position offset
        new_pos = (base_p + camera_offset).astype(np.float32)
        
        # Apply rotation offset (roll, pitch, yaw) to base quaternion
        if np.any(rotation_offset != 0):
            # Convert base quaternion (wxyz) to scipy format (xyzw)
            base_quat_xyzw = np.array([base_q[1], base_q[2], base_q[3], base_q[0]])
            base_rot = Rotation.from_quat(base_quat_xyzw)
            # Create offset rotation from euler angles (roll, pitch, yaw)
            offset_rot = Rotation.from_euler('xyz', rotation_offset)
            # Combine rotations: base * offset (local adjustment)
            combined_rot = base_rot * offset_rot
            # Convert back to wxyz format
            combined_quat_xyzw = combined_rot.as_quat()
            new_q = np.array([combined_quat_xyzw[3], combined_quat_xyzw[0], combined_quat_xyzw[1], combined_quat_xyzw[2]], dtype=np.float32)
        else:
            new_q = base_q.astype(np.float32)
        
        pose = sapien.Pose(p=new_pos, q=new_q)
    else:
        # Fall back to look_at behavior with config settings
        pos = sim_env.unwrapped.base_camera_settings["pos"] + camera_offset
        pose = sapien_utils.look_at(pos, sim_env.unwrapped.base_camera_settings["target"])
    
    sim_env.unwrapped.camera_mount.set_pose(pose)
    sim_env.unwrapped._sensors["base_camera"].camera.set_fovy(
        sim_env.unwrapped.base_camera_settings["fov"] + fov_offset
    )

    if len(active_keys) > 0:
        print("current_camera_position", pose.p)
        print("current_camera_quaternion", pose.q)
        print("rotation_offset (roll, pitch, yaw):", rotation_offset)
        print(
            "current_camera_fov",
            sim_env.unwrapped.base_camera_settings["fov"] + fov_offset,
        )
        help_message_printed = False  # Reset the flag when there's movement
    elif (
        not help_message_printed
    ):  # Only print help message if it hasn't been printed yet
        print("=== Commands for controlling sim camera ===")
        print("Position: (W/S) forward/back, (A/D) left/right, (UP/DOWN) up/down")
        print("Rotation: (Q/E) yaw, (R/F) pitch, (Z/X) roll")
        print("FOV: (LEFT/RIGHT) arrows")
        print("Reset: (BACKSPACE), Exit: close figure")
        print()
        help_message_printed = True

camera_offset = np.zeros(3, dtype=np.float32)
rotation_offset = np.zeros(3, dtype=np.float32)  # roll, pitch, yaw in radians
fov_offset = 0.0
active_keys = set()
last_frame_time = time.time()
MOVEMENT_SPEED = 0.1  # units per second
ROTATION_SPEED = 0.3  # radians per second
FOV_CHANGE_SPEED = 0.1  # radians per second
help_message_printed = False  # Flag to track if we've printed the help message
calibrated_base_pose = None  # Will be set if extrinsics path is provided


def on_key_press(event):
    global active_keys
    active_keys.add(event.key)


def on_key_release(event):
    global active_keys
    active_keys.discard(event.key)

def main(args: Args):
    real_robot = create_real_robot(uid="so100")
    
    # Check for existing motor calibration
    calibration_path = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so100_follower/stone_home.json"
    run_calibration = True
    if calibration_path.exists():
        print(f"\nExisting motor calibration found: {calibration_path}")
        response = input("Use existing calibration? [Y/n]: ").strip().lower()
        if response in ("", "y", "yes"):
            run_calibration = False
            print("Using existing calibration.")
        else:
            print("Will run new calibration...")
    
    real_robot.connect(calibrate=run_calibration)
    
    # If using existing calibration, write it to the motor hardware
    if not run_calibration and real_robot.calibration:
        real_robot.bus.write_calibration(real_robot.calibration)
        print("Calibration written to motor hardware.")
    
    real_agent = LeRobotRealAgent(real_robot)

    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="sensors",
        reward_mode="none",
        lighting_randomization={'enabled': False},
        distractor_objects={'enabled': False},
        # use larger camera resolution to make it easier to align. In training we won't use this however
        sensor_configs=dict(width=512, height=512)
    )
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))
    wrapper_kwargs = ['lighting_randomization', 'distractor_objects']
    env_kwargs = {k: v for k, v in env_kwargs.items() if k not in wrapper_kwargs}
            
    sim_env = gym.make(
        args.env_id,
        **env_kwargs,
    )
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent)
    # safety setup, now ctrl+c will first reset the robot to a resting position and then close environments and turn of torque
    setup_safe_exit(sim_env, real_env, real_agent)

    real_obs, _ = real_env.reset()

    # Determine which calibration to use
    global calibrated_base_pose
    extrinsics_path = None
    
    if args.extrinsics_path is not None:
        # Explicit path provided via CLI
        extrinsics_path = Path(args.extrinsics_path)
    elif args.use_existing_calibration:
        # Check for existing calibrations and prompt user
        calibrations = find_existing_calibrations(args.robot_id)
        if calibrations:
            extrinsics_path = prompt_for_calibration(calibrations)
        else:
            print(f"No existing calibrations found for robot '{args.robot_id}'")
    
    if extrinsics_path is not None:
        calibrated_base_pose = load_calibrated_pose(str(extrinsics_path))
        # Apply immediately so the first frame shows the calibrated view
        sim_env.unwrapped.camera_mount.set_pose(calibrated_base_pose)
        print(f"Loaded calibrated camera pose from {extrinsics_path}")
        print(f"  Position: {calibrated_base_pose.p}")
        print(f"  Quaternion (wxyz): {calibrated_base_pose.q}")
        print("Manual adjustment still available (WASD/arrows).")
    
    # Apply camera intrinsics if provided
    if args.intrinsics_path is not None:
        intrinsics = load_intrinsics(args.intrinsics_path)
        # Get the base_camera's render component
        camera_sensor = sim_env.unwrapped._sensors["base_camera"]
        apply_intrinsics_to_camera(camera_sensor.camera, intrinsics)
        print(f"Applied camera intrinsics from {args.intrinsics_path}")
        print(f"  fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
        print(f"  cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
        print(f"  Resolution: {intrinsics['width']}x{intrinsics['height']}")

    # for plotting robot camera reads
    fig = plt.figure()
    ax = fig.add_subplot()

    # Disable all default key bindings
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.manager.key_press_handler_id = None

    # initialize the plot
    im = ax.imshow(overlay_envs(sim_env, real_env))

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    print("Camera alignment: Move real camera to align with the sim camera, close figure to exit")
    while True:
        overlaid_imgs = overlay_envs(sim_env, real_env)
        im.set_data(overlaid_imgs)
        # Update camera position based on active keys
        update_camera(sim_env)
        # Redraw the plot
        fig.canvas.draw()
        fig.show()
        fig.canvas.flush_events()
        if not plt.fignum_exists(fig.number):
            print("The figure has been closed.")
            break

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)