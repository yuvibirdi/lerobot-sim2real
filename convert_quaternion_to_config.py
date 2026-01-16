#!/usr/bin/env python3
"""
Convert camera quaternion + position to ManiSkill camera config format.

This script helps convert quaternion-based camera poses to the target-based
format used in ManiSkill camera settings.
"""

import numpy as np
import json
from lerobot_sim2real.utils.camera_calibration import quaternion_to_target


def main():
    # Your data from camera alignment
    current_camera_position = np.array([0.00258458, 0.03813902, 0.4233188])
    current_camera_quaternion = np.array([0.97272766, -0.0295315, 0.22976297, 0.01173956])  # wxyz format
    current_camera_fov = 1.3277327966690062
    
    # Compute distance from existing config to maintain similar viewing distance
    # You can adjust this distance based on your needs
    existing_pos = np.array([0.002584, 0.038139, 0.4233188])
    existing_target = np.array([0.5, 0.16, 0.052])
    distance = np.linalg.norm(existing_target - existing_pos)
    
    print(f"Using distance: {distance:.6f} (computed from existing config)")
    print()
    
    # Convert quaternion to target
    target = quaternion_to_target(current_camera_position, current_camera_quaternion, distance=distance)
    
    # Create config
    config = {
        "base_camera_settings": {
            "pos": current_camera_position.tolist(),
            "fov": float(current_camera_fov),
            "target": target.tolist()
        }
    }
    
    print("=== Generated Config ===")
    print(json.dumps(config, indent=2))
    print()
    
    # Also print in a format that's easy to copy
    print("=== Easy Copy Format ===")
    print(f'"pos": {current_camera_position.tolist()},')
    print(f'"fov": {current_camera_fov},')
    print(f'"target": {target.tolist()}')


if __name__ == "__main__":
    main()
