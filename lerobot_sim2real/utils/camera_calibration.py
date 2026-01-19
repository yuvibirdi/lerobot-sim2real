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
    
    NOTE: This helper keeps the historical behavior of producing a target at a fixed
    distance along the camera's viewing direction. For sim2real tabletop camera
    configs you usually want a geometric target, e.g. intersection with the table
    plane (z=0). Prefer `quaternion_to_table_intersection_target`.
    
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
    
    forward_world = camera_forward_world_from_quaternion(quaternion)
    
    # Compute target point
    target = position + forward_world * distance
    
    return target.astype(np.float32)


def camera_up_world_from_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """
    Compute the camera up direction in world coordinates from a camera pose quaternion.

    Convention used throughout this repo:
    - Quaternions are **wxyz** (SAPIEN Pose `.q`).
    - Camera local axes (SAPIEN convention): +X is forward, +Y is left, **+Z is up**.

    Args:
        quaternion_wxyz: Quaternion [w, x, y, z]

    Returns:
        Unit 3-vector in world frame pointing along the camera's up direction.
    """
    q = np.array(quaternion_wxyz, dtype=np.float64).flatten()
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    # wxyz -> xyzw (scipy)
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    rot = Rotation.from_quat(quat_xyzw)
    # Camera up is +Z in camera local space (SAPIEN convention)
    up_world = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    n = np.linalg.norm(up_world)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (up_world / n).astype(np.float32)


def camera_forward_world_from_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """
    Compute the camera viewing direction in world coordinates from a camera pose quaternion.

    Convention used throughout this repo:
    - Quaternions are **wxyz** (SAPIEN Pose `.q`).
    - Camera local axes follow the common robotics convention: **+X is forward**.

    Args:
        quaternion_wxyz: Quaternion [w, x, y, z]

    Returns:
        Unit 3-vector in world frame pointing along the camera's forward/view direction.
    """
    q = np.array(quaternion_wxyz, dtype=np.float64).flatten()
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    # wxyz -> xyzw (scipy)
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    rot = Rotation.from_quat(quat_xyzw)
    forward_world = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    n = np.linalg.norm(forward_world)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (forward_world / n).astype(np.float32)


def quaternion_to_table_intersection_target(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    table_z: float = 0.0,
    forward_z_epsilon: float = 1e-6,
    fallback_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert (position, quaternion) to a ManiSkill-style `target` by intersecting the
    camera view ray with the table plane z=table_z.

    This avoids arbitrary “pick a distance” hacks and produces a geometrically
    meaningful target point for tabletop tasks (ManiSkill convention: z=0 is the table surface).

    Args:
        position: Camera position [x, y, z] (numpy or torch tensor)
        quaternion_wxyz: Camera quaternion [w, x, y, z] (numpy or torch tensor)
        table_z: Table plane height in world coordinates
        forward_z_epsilon: Treat |forward.z| < eps as parallel to plane
        fallback_xy: (x, y) to use if the ray does not hit the plane in front of the camera

    Returns:
        Target [x, y, z] on the table plane.
    """
    # Convert tensors to numpy arrays if needed
    if hasattr(position, "cpu"):
        position = position.cpu().numpy()
    position = np.array(position, dtype=np.float64).flatten()
    if position.shape[0] != 3:
        raise ValueError(f"Expected position shape (3,), got {position.shape}")

    if hasattr(quaternion_wxyz, "cpu"):
        quaternion_wxyz = quaternion_wxyz.cpu().numpy()
    quaternion_wxyz = np.array(quaternion_wxyz, dtype=np.float64).flatten()

    forward = camera_forward_world_from_quaternion(quaternion_wxyz).astype(np.float64)
    fz = float(forward[2])
    # Ray: p(t) = position + t * forward, want pz(t) = table_z
    if abs(fz) > forward_z_epsilon:
        t = (float(table_z) - float(position[2])) / fz
        if t > 0.0:
            hit = position + t * forward
            hit[2] = float(table_z)
            return hit.astype(np.float32)

    # Fallback: no hit in front of camera (ray points upward / parallel / behind)
    return np.array([float(fallback_xy[0]), float(fallback_xy[1]), float(table_z)], dtype=np.float32)


def patch_camera_pose_from_quaternion(sim_env):
    """
    Monkey-patch the environment's sample_camera_poses to use quaternion from config.
    
    If base_camera_settings contains a "quaternion" key, this function patches
    sample_camera_poses to use the full quaternion pose (preserving roll) instead
    of the default look_at(pos, target) which loses roll.
    
    This approach keeps all customization in lerobot_sim2real without modifying ManiSkill.
    
    Args:
        sim_env: ManiSkill environment (wrapped or unwrapped)
        
    Returns:
        True if patch was applied, False if no quaternion in config
    """
    env = sim_env.unwrapped
    
    # Check if quaternion is in config
    if "quaternion" not in env.base_camera_settings:
        return False
    
    quaternion_wxyz = np.array(env.base_camera_settings["quaternion"], dtype=np.float32)
    pos = np.array(env.base_camera_settings["pos"], dtype=np.float32)
    
    # Validate
    if quaternion_wxyz.shape != (4,):
        raise ValueError(f"quaternion must be [w,x,y,z] shape (4,), got {quaternion_wxyz.shape}")
    if pos.shape != (3,):
        raise ValueError(f"pos must be [x,y,z] shape (3,), got {pos.shape}")
    
    # Create the patched method
    def quaternion_sample_camera_poses(n: int):
        """Sample camera poses using quaternion directly (preserves roll)."""
        # Create pose from position and quaternion
        return sapien.Pose(p=pos, q=quaternion_wxyz)
    
    # Apply the patch
    env.sample_camera_poses = quaternion_sample_camera_poses
    
    print(f"Patched camera pose to use quaternion from config:")
    print(f"  Position: {pos.tolist()}")
    print(f"  Quaternion (wxyz): {quaternion_wxyz.tolist()}")
    
    # Also apply immediately to current camera mount
    env.camera_mount.set_pose(sapien.Pose(p=pos, q=quaternion_wxyz))
    
    return True

