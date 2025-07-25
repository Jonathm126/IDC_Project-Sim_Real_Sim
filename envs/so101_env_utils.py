from pathlib import Path

DT = 0.034  # 0.002 ms physics * 17 = 30 hz

JOINTS = [
        # absolute joint position
        'shoulder_pan',
        'shoulder_lift',
        'elbow_flex',
        'wrist_flex',
        'wrist_roll',
        # normalized gripper position 0: close, 1: open
        'gripper']

ACTIONS = [
        # absolute joint position
        'shoulder_pan',
        'shoulder_lift',
        'elbow_flex',
        'wrist_flex',
        'wrist_roll',
        # normalized gripper position 0: close, 1: open
        'gripper']

START_ARM_POSE = [0] *6 #TODO

JOINTS_MAX = 2
JOINTS_MIN = -2

MUJOCO_DIR = Path(__file__).parent.resolve() / "mujoco" 

MASTER_GRIPPER_JOINT_OPEN = 1.75
MASTER_GRIPPER_JOINT_CLOSE = -0.174
MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2

def normalize_master_gripper_position(x):
        return (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)

def unnormalize_master_gripper_position(x): 
        return x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE

SO101TASKS = ["TableLegAssembleTask", "TableLegMoveTask"]
SO101OBSTYPES = ["pixels", "pixels_agent_pos", "pixels_agent_pos_state"]