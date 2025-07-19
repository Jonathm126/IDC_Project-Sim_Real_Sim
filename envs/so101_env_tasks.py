import collections

import numpy as np
from dm_control.suite import base

class SO101Task(base.Task,):
    def __init__(self, random=None, observation_height = 480, observation_width = 640):
        self.observation_height = observation_height
        self.observation_width = observation_width
        super().__init__(random=random)
    
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)
    
    def before_step(self, action, physics):
        super().before_step(action, physics)
    
    @staticmethod
    def get_qpos(physics):
        return None
    
    @staticmethod
    def get_qvel(physics):
        return None
    
    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError
    
    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        # obs["env_state"] = self.get_env_state(physics) TODO temporarys
        obs["images"] = {}
        obs["images"]["top_cam"]   = physics.render(height=self.observation_height, width=self.observation_width, camera_id="top_cam")
        obs["images"]["wrist_cam"] = physics.render(height=self.observation_height, width=self.observation_width, camera_id="wrist_cam")
        obs["images"]["iso_cam"]   = physics.render(height=self.observation_height, width=self.observation_width, camera_id="iso_cam")
        
        return obs
        
    def get_reward(self, physics):
        return 0

class TableLegMoveTask(SO101Task):
    def __init__(self, random = None, observation_height = 480, observation_width = 640):
        super().__init__(random, observation_height, observation_width)