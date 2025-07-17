import time
import numpy as np

def move_robot_to_pose(robot, target_pose, duration_sec=1.0, fps=30):
    dt = 1.0 / fps
    steps = max(1, int(duration_sec * fps))

    # Get joint keys and current values from observation
    obs = robot.get_observation()
    joint_keys = [key for key in obs if key.endswith(".pos")]
    current_pose = [obs[key] for key in joint_keys]

    # Ensure target_pose is in the same order
    if isinstance(target_pose, dict):
        target_pose = [target_pose[key] for key in joint_keys]

    current_pose = np.array(current_pose, dtype=np.float32)
    target_pose = np.array(target_pose, dtype=np.float32)

    # Interpolate and send action
    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated_pose = (1 - alpha) * current_pose + alpha * target_pose
        action_dict = dict(zip(joint_keys, interpolated_pose))
        robot.send_action(action_dict)
        time.sleep(dt)
