# this file contains the classes neede to register the SO101Env robot environment with the LeRobot
# framework, by providing a config that works with LeRobot's scripts

from dataclasses import dataclass, field
import numpy as np

# lerobot
from lerobot.envs.configs import EnvConfig
from lerobot.scripts.rl.gym_manipulator import ConvertToLeRobotObservation, BatchCompatibleWrapper, TorchActionWrapper, TimeLimitWrapper

# lerobot contsts
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

# my code
from envs.so101_env_utils import SO101TASKS, SO101OBSTYPES
from envs.so101_env import SO101Env


@dataclass(kw_only=True)
# @EnvConfig.register_subclass("so101")
class SO101EnvConfig(EnvConfig):
    """Parametrers for the SO101Env:
        Task: can be one of the SO101Tasks
        obs_type: can be one of the SO101ObsTypes
        episode_length: in steps
        control_time_s: optional, time limit in sec
        reset_time_s: optional, reset time between episodes in sec
        reset_pose: optional, pose to reset to
    """
    # define general config params
    task          : str
    device        : str
    obs_type      : str
    fps           : int  = 30
    episode_length: int  = 400
    control_time_s: int | None = None
    reset_time_s  : int | None = None
    reset_pose    : np.ndarray | None = None
    render_mode   : str = "rgb_array"
    observation_width    :int = 640
    observation_height   :int = 480
    visualization_width  :int = 640
    visualization_height :int = 480
    
    # the minimal feature is the action
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))
        }
    )
    
    # map between dataset features and env (gym) features
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action"          : ACTION,
            "agent_pos"       : OBS_STATE,
            "env_state"       : OBS_ENV_STATE,
            "top_cam"         : f"{OBS_IMAGE}.top_cam",
            "pixels/top_cam"  : f"{OBS_IMAGES}.top_cam",
            "wrist_cam"       : f"{OBS_IMAGE}.wrist_cam",
            "pixels/wrist_cam": f"{OBS_IMAGES}.wrist_cam",
        }
    )
    
    # add logic post init for validation + addition
    def __post_init__(self):
        # validate
        if self.obs_type not in SO101OBSTYPES or self.task not in SO101TASKS:
            raise ValueError('Bad Observation type or task')
        
        # add image features
        if self.obs_type in["pixels", "pixels_agent_pos", "pixels_agent_pos_state"]:
            self.features["pixels/top_cam"] = PolicyFeature(type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3))
            self.features["pixels/wrist_cam"] = PolicyFeature(type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3))
        
        # add robot state feature
        if self.obs_type in["pixels_agent_pos"]:
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
        
        # add env state feature   
        if self.obs_type in["pixels_agent_pos_state"]:
            raise NotImplementedError

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }

def make_so101_env(cfg: SO101EnvConfig, torch_actions):
    """ Builds SO101Env from cfg."""
    env = SO101Env(
        task                 = cfg.task,
        obs_type             = cfg.obs_type,
        render_mode          = cfg.render_mode,
        observation_width    = cfg.observation_width,
        observation_height   = cfg.observation_height,
        visualization_width  = cfg.visualization_width,
        visualization_height = cfg.visualization_height
    )
    
    # convert to lerobot policy structure
    env = ConvertToLeRobotObservation(env, cfg.device)
    
    # add wrappers to convert to torch batches
    env = BatchCompatibleWrapper(env)
    if torch_actions:
        env = TorchActionWrapper(env, device = cfg.device)  
    
    # time limiter
    if cfg.control_time_s is not None:
        env = TimeLimitWrapper(env, cfg.control_time_s, cfg.fps)
    
    return env