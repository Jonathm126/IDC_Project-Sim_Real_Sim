from pathlib import Path

FPS = 30
DT = 1/FPS  # 0.02 ms -> 1/0.2 = 50 hz

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

START_ARM_POSE = [0] *7

ASSETS_DIR = Path(__file__).parent.resolve() / "assets" 

MASTER_GRIPPER_JOINT_OPEN = 1.75
MASTER_GRIPPER_JOINT_CLOSE = -0.174
MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2

def normalize_master_gripper_position(x):
        return (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)

def unnormalize_master_gripper_position(x): 
        return x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE

