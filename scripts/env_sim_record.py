
from dataclasses import dataclass, field
import numpy as np

# lerobot
from lerobot.envs.configs import EnvConfig
from lerobot.scripts.rl.gym_manipulator import ConvertToLeRobotObservation, ResetWrapper, BatchCompatibleWrapper, TorchActionWrapper

# lerobot contsts
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

# my code
from envs.so101_env_utils import SO101TASKS, SO101OBSTYPES
from envs.so101_env import SO101Env
from envs.so101_env_config import SO101EnvConfig

def record_dataset(env, policy, cfg):
