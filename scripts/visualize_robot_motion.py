# IMPORTANT: Set rendering backend BEFORE any imports that use OpenGL/mujoco
# This must be at the very top of the file
import sys
import os

# Check if --headless is in command line args BEFORE importing mujoco
if '--headless' in sys.argv:
    # Use EGL for headless rendering (osmesa is fallback)
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import pickle
import time
import numpy as np
import pathlib

# Add the parent directory to sys.path to allow importing modules from the parent directory
HERE = pathlib.Path(__file__).parent
sys.path.append(str(HERE.parent))

from general_motion_retargeting import RobotMotionViewer
from rich import print

def visualize_motion(motion_file, robot_type, record_video=False, loop=False, headless=False):
    print(f"Loading motion from {motion_file}")
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)

    fps = motion_data.get("fps", 30)
    root_pos = motion_data["root_pos"]
    root_rot_xyzw = motion_data["root_rot"]
    dof_pos = motion_data["dof_pos"]

    # Convert quaternions from xyzw (pickle format) to wxyz (mujoco format)
    # xyzw -> [3, 0, 1, 2] -> wxyz
    # But wait, smplx_to_robot.py does: qpos[3:7][[1,2,3,0]] to save as xyzw.
    # original qpos is [w, x, y, z].
    # saved is [x, y, z, w].
    # To restore [w, x, y, z]:
    # take saved [x, y, z, w]
    # new[0] = saved[3] (w)
    # new[1] = saved[0] (x)
    # new[2] = saved[1] (y)
    # new[3] = saved[2] (z)
    
    root_rot_wxyz = np.zeros_like(root_rot_xyzw)
    root_rot_wxyz[:, 0] = root_rot_xyzw[:, 3] # w
    root_rot_wxyz[:, 1] = root_rot_xyzw[:, 0] # x
    root_rot_wxyz[:, 2] = root_rot_xyzw[:, 1] # y
    root_rot_wxyz[:, 3] = root_rot_xyzw[:, 2] # z
    
    num_frames = root_pos.shape[0]
    print(f"Loaded {num_frames} frames, FPS: {fps}")

    video_path = None
    # In headless mode, always record video (otherwise nothing is displayed)
    if headless:
        record_video = True
    if record_video:
        motion_name = pathlib.Path(motion_file).stem
        video_path = f"videos/viz_{robot_type}_{motion_name}.mp4"

    viewer = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=fps,
        record_video=record_video,
        video_path=video_path,
        camera_follow=True,
        headless=headless,
    )

    i = 0
    try:
        while True:
            if loop:
                frame_idx = i % num_frames
            else:
                frame_idx = i
                if frame_idx >= num_frames:
                    break
            
            viewer.step(
                root_pos=root_pos[frame_idx],
                root_rot=root_rot_wxyz[frame_idx],
                dof_pos=dof_pos[frame_idx],
                follow_camera=True
            )
            i += 1
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        viewer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize robot motion from .pkl file")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to the .pkl motion file")
    parser.add_argument(
        "--robot",
        default="unitree_g1",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        help="Robot type"
    )
    parser.add_argument("--record_video", action="store_true", help="Record video")
    parser.add_argument("--loop", action="store_true", help="Loop the animation")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no window, output video only)")

    args = parser.parse_args()

    visualize_motion(args.motion_file, args.robot, args.record_video, args.loop, args.headless)
