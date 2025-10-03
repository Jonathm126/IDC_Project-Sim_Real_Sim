from pathlib import Path

# paths
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from robot.robot_config import robot

robot.connect()
obs = robot.get_observation()
keys = robot.action_features.keys()
pose = {k:obs[k] for k in keys}
print(pose)
robot.disconnect()
