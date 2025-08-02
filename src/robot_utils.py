import time
import numpy as np
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.teleoperators import Teleoperator


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


def norm_modes_to_ranges(teleop:Teleoperator):
    ''' Get the teleoperator's joint ranges as a list of min and max'''
    norm_modes = [motor.norm_mode for _,motor in teleop.bus.motors.items()]
    ranges = []
    for mode in norm_modes:
        if mode == MotorNormMode.RANGE_M100_100:
            ranges.append((-100.0, 100.0))
        elif mode == MotorNormMode.RANGE_0_100:
            ranges.append((0.0, 100.0))
        else:
            raise ValueError(f"Unknown norm_mode: {mode}")
    return np.array(ranges, dtype=np.float32)

class JointSpaceNormalizer:
    """Maps between MuJoCo joint space and real robot joint space using calibration."""

    def __init__(self, teleop, mujoco_ranges):
        # Load teleop calibration
        calib = teleop.calibration
        calib_ranges = np.array([
            (cal.range_min, cal.range_max, cal.homing_offset)
            for cal in calib.values()
        ], dtype=np.float32)

        self.robot_min = calib_ranges[:, 0]
        self.robot_max = calib_ranges[:, 1]
        self.offsets   = calib_ranges[:, 2]

        # Load MuJoCo joint limits
        self.mujoco_min = mujoco_ranges[:, 0]
        self.mujoco_max = mujoco_ranges[:, 1]

    def mujoco_to_robot(self, mj_qpos: np.ndarray) -> np.ndarray:
        """Convert MuJoCo joint positions -> robot joint positions."""
        normed = (mj_qpos - self.mujoco_min) / (self.mujoco_max - self.mujoco_min)
        robot_vals = self.robot_min + normed * (self.robot_max - self.robot_min)
        return robot_vals + self.offsets

    def robot_to_mujoco(self, robot_qpos: np.ndarray) -> np.ndarray:
        """Convert robot joint positions -> MuJoCo joint positions."""
        raw = robot_qpos - self.offsets
        normed = (raw - self.robot_min) / (self.robot_max - self.robot_min)
        mj_qpos = self.mujoco_min + normed * (self.mujoco_max - self.mujoco_min)
        return mj_qpos

    def check_round_trip(self, qpos: np.ndarray) -> float:
        """Return max error after mujoco->robot->mujoco round trip."""
        back = self.robot_to_mujoco(self.mujoco_to_robot(qpos))
        return float(np.abs(qpos - back).max())

def synthetic_leader_dataset(leader_ranges, n_steps=500, seed=0):
    """
    Generate synthetic leader joint positions using the leader's normalized ranges.

    Args:
        leader_ranges: a list of tuples which are the range per joint
        n_steps: number of steps to generate
        seed: RNG seed for reproducibility
    Returns:
        np.ndarray of shape (n_steps, n_joints)
    """
    rng = np.random.default_rng(seed)

    # Use normalization ranges (e.g., [-100,100] or [0,100])
    lows, highs = np.array([r[0] for r in leader_ranges], dtype=np.float32), \
                np.array([r[1] for r in leader_ranges], dtype=np.float32)

    t = np.linspace(0, 2 * np.pi, n_steps)
    synthetic = []
    for i, (low, high) in enumerate(zip(lows, highs)):
        amp = (high - low) / 2
        mid = (high + low) / 2
        vals = mid + amp * np.sin(t + rng.uniform(0, 2*np.pi))
        vals += rng.normal(0, amp * 0.05, size=n_steps)  # add small noise
        synthetic.append(vals)

    return np.stack(synthetic, axis=1)  # (n_steps, n_joints)

def sweep_leader_dataset(leader_ranges, steps_per_axis=200, center_pause_steps=20):
    """
    Generate a sweep dataset where each joint is moved from min->max and back to center,
    one axis at a time, using the leader's normalized ranges.

    Args:
        leader_ranges: a list of tuples which are the range per joint
        steps_per_axis: number of steps to go from min to max for each axis
        center_pause_steps: how many steps to hold at the center before switching axis
    Returns:
        np.ndarray of shape (n_steps_total, n_joints)
    """
    joint_mins = np.array([low for (low, _) in leader_ranges], dtype=np.float32)
    joint_maxs = np.array([high for (_, high) in leader_ranges], dtype=np.float32)

    centers = (joint_mins + joint_maxs) / 2
    n_joints = len(joint_mins)
    dataset = []

    for j in range(n_joints):
        # Start at center
        pose = centers.copy()
        dataset.extend([pose.copy()] * center_pause_steps)

        # Sweep from min → max
        sweep_up = np.linspace(joint_mins[j], joint_maxs[j], steps_per_axis)
        for val in sweep_up:
            pose = centers.copy()
            pose[j] = val
            dataset.append(pose.copy())

        # Return to center
        dataset.extend([centers.copy()] * center_pause_steps)

    return np.stack(dataset, axis=0)  # (n_steps_total, n_joints)
