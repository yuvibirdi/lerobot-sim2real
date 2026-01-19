from pathlib import Path
import gymnasium as gym
from lerobot.common.robots.robot import Robot
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.robots.utils import make_robot_from_config
import numpy as np
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from lerobot_sim2real.utils.platform import is_macos
from lerobot_sim2real.utils.macos_camera import MacOSStereoCamera
from lerobot_sim2real.utils.macos_camera_config import MacOSStereoCameraConfig


def create_macos_stereo_camera(
    index: int = 0,
    stereo_side: str = "right",
    fps: int = 30,
) -> MacOSStereoCamera:
    """
    Create a macOS stereo camera for use as base_camera.
    
    Args:
        index: Camera index (0, 1, 2, ...)
        stereo_side: "left" or "right" half of stereo frame
        fps: Target frames per second
        
    Returns:
        Connected MacOSStereoCamera instance
    """
    config = MacOSStereoCameraConfig(
        index=index,
        fps=fps,
        stereo_side=stereo_side,
        capture_width=2560,
        capture_height=720,
        output_width=1280,
        output_height=720,
    )
    camera = MacOSStereoCamera(config)
    return camera


def create_real_robot(uid: str = "so100") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file.
    
    On macOS with stereo cameras, we create the robot with NO cameras initially,
    then inject our custom MacOSStereoCamera after creation. This bypasses lerobot's
    strict resolution validation that doesn't work with 2560x720 stereo cameras.
    """
    if uid == "so100":
        # Platform-specific camera configuration
        if is_macos():
            # macOS: Create robot without cameras, inject our camera after
            cameras = {}
        else:
            # Linux/Windows: Use standard OpenCV camera
            cameras = {
                "base_camera": OpenCVCameraConfig(index_or_path=0, fps=30, width=1280, height=720)
            }
        
        robot_config = SO100FollowerConfig(
            # port="/dev/ttyACM0",
            port="/dev/tty.usbmodem58FA0926321",
            use_degrees=True,
            cameras=cameras,
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            # cameras={
            #     "base_camera": RealSenseCameraConfig(serial_number_or_name="146322070293", fps=30, width=640, height=480)
            # },
            id="stone_home",
        )
        real_robot = make_robot_from_config(robot_config)
        return real_robot