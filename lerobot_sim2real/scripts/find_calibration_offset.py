"""
Find the correct CALIBRATION_OFFSET for EasyHEC.

This script helps you determine the offset between:
- What LeRobot motor drivers report
- What the URDF expects

Run this and manually move your robot to a known pose (like "home" or "zero"),
then see what values are reported.
"""
import numpy as np
from dataclasses import dataclass
import tyro
import time
import json
from pathlib import Path
from lerobot_sim2real.config.real_robot import create_real_robot


@dataclass
class Args:
    robot_id: str = "stone_home"
    calibration_file: str = ""  # Optional: path to lerobot calibration json


def main(args: Args):
    print("=" * 70)
    print("CALIBRATION OFFSET FINDER FOR EASYHEC")
    print("=" * 70)
    
    # Try to load the lerobot calibration file
    cal_path = Path(f"~/.cache/huggingface/lerobot/calibration/robots/so100_follower/{args.robot_id}.json").expanduser()
    if args.calibration_file:
        cal_path = Path(args.calibration_file)
    
    if cal_path.exists():
        print(f"\nFound LeRobot calibration file: {cal_path}")
        with open(cal_path, 'r') as f:
            cal_data = json.load(f)
        
        print("\n--- LeRobot Calibration Data ---")
        if "homing_offset" in cal_data:
            print("\nhoming_offset (these might be the values you need):")
            for name, val in cal_data["homing_offset"].items():
                print(f"  {name:15s}: {val}")
        
        if "motor_names" in cal_data:
            print(f"\nMotor names: {cal_data['motor_names']}")
        
        print()
    else:
        print(f"\nNo calibration file found at: {cal_path}")
    
    # Connect to robot
    print("\nConnecting to robot...")
    robot = create_real_robot(uid="so100")
    robot.connect()
    
    print("\n" + "=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print("""
1. PHYSICALLY move your robot to a KNOWN pose (e.g., all joints at 0 degrees)
2. Look at what values the robot reports below
3. The CALIBRATION_OFFSET should be calculated as:
   
   For EasyHEC's so100.py, it uses motor steps not radians.
   The offset converts from motor steps to URDF radians.
   
   Check the easyhec code to understand the exact conversion.
""")
    
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    print("\n--- Live Joint Readings (press Ctrl+C to stop) ---\n")
    
    try:
        while True:
            obs = robot.get_observation()
            
            print("\rJoint positions: ", end="")
            for name in joint_names:
                if name in obs:
                    val = obs[name]
                    print(f"{name[:3]}={np.degrees(val):+6.1f}° ", end="")
            print("   (move robot, values update)", end="", flush=True)
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    
    # Get final reading
    obs = robot.get_observation()
    print("\n--- Final Joint Positions ---")
    for name in joint_names:
        if name in obs:
            val = obs[name]
            print(f"  {name:15s}: {val:+8.4f} rad = {np.degrees(val):+8.2f} deg")
    
    robot.disconnect()
    real_env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))

