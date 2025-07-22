# gym
import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

# my constants
from so101_env_utils import ACTIONS, DT, JOINTS, MUJOCO_DIR

# my tasks
from so101_env_tasks import TableLegMoveTask

#gym_aloha 
from gym_aloha.env import AlohaEnv
from gym_aloha.utils import sample_box_pose, sample_insertion_pose

class SO101Env(gym.Env):
    # this is a master environemnt for the SO101 tasks
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        task    : str,
        obs_type: str,
        render_mode          = "rgb_array",
        observation_width    = 640,
        observation_height   = 480,
        visualization_width  = 640,
        visualization_height = 480,
        ):
        ''' Builds the SO101 gym env.
            task = 'TableLegMoveTask', 'TableLegAssembleTask'
            obs_type = 'pixels': only images, 'pixels_agent_pos': images and robot pose, 'pixels_agent_pos_state': privleged state incl. extra positions
            render_mode must be 'rgb_array'
        '''
        
        # build the env super class
        super().__init__()
        self.task                 = task
        self.obs_type             = obs_type
        self.render_mode          = render_mode
        self.observation_width    = observation_width
        self.observation_height   = observation_height
        self.visualization_width  = visualization_width
        self.visualization_height = visualization_height
        
        # make gym env and task
        self._env = self._make_env_task(self.task)
        
        # build the observation vector based on the different observation modes
        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top_cam": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist_cam": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )
        
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top_cam": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist_cam": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )
        
        elif self.obs_type == "pixels_agent_pos_state":
            raise NotImplementedError()
        
        else:
            raise NotImplementedError()
        
        # define action space TODO why not the same as obs space?
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)
    
    def _make_env_task(self, task_name):
        # build the env according to the task type
        # table leg moving task
        if task_name == 'TableLegMoveTask':
            xml_path = MUJOCO_DIR / 'so101_table_leg_move.xml'
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TableLegMoveTask(observation_height = self.observation_height,
                                    observation_width = self.observation_width)
        else:
            raise NotImplementedError()
        
        # finally build the env itself, time limit is sent to inf? TODO
        env = control.Environment(physics, task, float("inf"), control_timestep=DT, n_sub_steps=None, flat_observation=False)
        
        return env
    
    def _format_raw_obs(self, raw_obs):
        # for pixels only return the images
        if self.obs_type == "pixels":
            obs = {
                "pixels": {
                    "top_cam": raw_obs["images"]["top_cam"].copy(),
                    "wrist_cam": raw_obs["images"]["wrist_cam"].copy(),
                },
            }
        # for pixels and agent state only return the images and agent pos
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {
                    "top_cam": raw_obs["images"]["top_cam"].copy(),
                    "wrist_cam": raw_obs["images"]["wrist_cam"].copy(),
                },
                "agent_pos": raw_obs["qpos"],
            }
        elif self.obs_type == "pixels_agent_pos_state":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return obs
    
    def step(self, action):
        # make sure the action is one time step
        assert action.ndim == 1
        # make step TODO are we sure about the order of returns
        _, reward, _, raw_obs = self._env.step(action)  
        
        terminated = is_success = reward == 10
        info = {"is_success": is_success}
        
        observation = self._format_raw_obs(raw_obs)
        
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        image = self._env.physics.render(height=height, width=width, camera_id="top_cam")
        return image
    
    def reset(self, seed=None, options=None):
        raise NotImplementedError
    
    def close(self):
        pass