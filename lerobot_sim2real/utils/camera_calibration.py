"""
Utilities for loading and applying camera calibration from easyhec.

Uses calibrated position with look_at for initial orientation.
User can fine-tune rotation with Q/E/R/F/Z/X controls.
"""

import json
import numpy as np
import sapien
from mani_skill.utils import sapien_utils
from scipy.spatial.transform import Rotation


def load_calibrated_pose(extrinsics_path: str):
    """
    Load easyhec camera extrinsics and return as SAPIEN pose.
    
    Uses the calibrated position and look_at for orientation.
    
    Args:
        extrinsics_path: Path to camera_extrinsic_ros.npy from easyhec calibration
        
    Returns:
        SAPIEN Pose object
    """
    extrinsic = np.load(extrinsics_path)
    
    # Position is correct in ROS extrinsic: X forward, Y left, Z up
    position = extrinsic[:3, 3].astype(np.float32)
    
    # Use look_at pointing at robot workspace
    target = np.array([0.3, 0.0, 0.05], dtype=np.float32)
    
    return sapien_utils.look_at(position, target)


def load_intrinsics(intrinsics_path: str) -> dict:
    """
    Load camera intrinsics from JSON file.
    
    Args:
        intrinsics_path: Path to camera_intrinsic.json
        
    Returns:
        Dictionary with fx, fy, cx, cy, width, height
    """
    with open(intrinsics_path, "r") as f:
        return json.load(f)


def apply_intrinsics_to_camera(camera, intrinsics: dict, near: float = 0.01, far: float = 10.0):
    """
    Apply calibrated intrinsics to a SAPIEN camera using set_perspective_parameters.
    
    Args:
        camera: SAPIEN camera object (RenderCameraComponent)
        intrinsics: Dictionary with fx, fy, cx, cy
        near: Near clipping plane
        far: Far clipping plane
    """
    camera.set_perspective_parameters(
        near=near,
        far=far,
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        skew=0.0
    )


def apply_calibrated_camera_pose(sim_env, extrinsics_path: str):
    """
    Load easyhec camera extrinsics and set the simulation camera pose.
    
    Args:
        sim_env: ManiSkill simulation environment (unwrapped or wrapped)
        extrinsics_path: Path to camera_extrinsic_ros.npy from easyhec calibration
    """
    pose = load_calibrated_pose(extrinsics_path)
    sim_env.unwrapped.camera_mount.set_pose(pose)
    print(f"Applied calibrated camera pose from {extrinsics_path}")
    print(f"  Position: {pose.p}")
    print(f"  Quaternion (wxyz): {pose.q}")


def quaternion_to_target(position: np.ndarray, quaternion: np.ndarray, distance: float = 0.5) -> np.ndarray:
    """
    Convert camera position and quaternion to a target point for ManiSkill config.
    
    In SAPIEN/ROS convention, cameras look down the negative Z axis in camera space.
    This function computes where the camera is looking at in world space.
    
    Args:
        position: Camera position [x, y, z] in world space (can be tensor or numpy array)
        quaternion: Camera orientation quaternion in wxyz format [w, x, y, z] (can be tensor or numpy array)
        distance: Distance from camera to target point (default: 0.5)
        
    Returns:
        Target point [x, y, z] that the camera is looking at
    """
    # Convert tensors to numpy arrays if needed
    if hasattr(position, 'cpu'):
        position = position.cpu().numpy()
    if hasattr(position, 'flatten'):
        position = position.flatten()
    position = np.array(position)
    
    if hasattr(quaternion, 'cpu'):
        quaternion = quaternion.cpu().numpy()
    if hasattr(quaternion, 'flatten'):
        quaternion = quaternion.flatten()
    quaternion = np.array(quaternion)
    
    # Ensure we have a 1D array
    if quaternion.ndim > 1:
        quaternion = quaternion[0] if quaternion.shape[0] == 1 else quaternion.flatten()
    if position.ndim > 1:
        position = position[0] if position.shape[0] == 1 else position.flatten()
    
    # Convert quaternion from wxyz to xyzw format for scipy
    quat_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    
    # Create rotation object
    rot = Rotation.from_quat(quat_xyzw)
    
    # Camera forward direction in camera space (ROS convention: X forward, Y left, Z up)
    # SAPIEN cameras look along the positive X axis in camera space
    forward_camera_space = np.array([1.0, 0.0, 0.0])
    
    # Rotate forward direction to world space
    forward_world = rot.apply(forward_camera_space)
    
    # Compute target point
    target = position + forward_world * distance
    
    return target.astype(np.float32)

