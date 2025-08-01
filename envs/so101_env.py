# gym
import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

# my constants
from envs.so101_env_utils import ACTIONS, DT, JOINTS, MUJOCO_DIR, JOINTS_MAX, JOINTS_MIN, SO101OBSTYPES, SO101TASKS

# my tasks
from envs.so101_env_tasks import TableLegAssembleTask

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
        
        # make sure the observation type is legal
        assert self.obs_type in SO101OBSTYPES, f"Invalid obs_type: {self.obs_type}"
        assert self.task in SO101TASKS, f"Invalid task: {self.task}"
        
        # build the observation vector based on the different observation modes
        if self.obs_type == "pixels" or "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top_cam": spaces.Box(
                                low   = 0,
                                high  = 255,
                                shape = (self.observation_height, self.observation_width, 3),
                                dtype = np.uint8,
                            ),
                            "wrist_cam": spaces.Box(
                                low   = 0,
                                high  = 255,
                                shape = (self.observation_height, self.observation_width, 3),
                                dtype = np.uint8,
                            ),
                        }
                    ),
                }
            )
        
        if self.obs_type == "pixels_agent_pos":
            self.observation_space["agent_pos"] = spaces.Box( 
                                                low   = JOINTS_MIN,
                                                high  = JOINTS_MAX,
                                                shape = (len(JOINTS),),
                                                dtype = np.float64,
                                            )
        
        if self.obs_type == "pixels_agent_pos_state":
            raise NotImplementedError()
        
        # define action space TODO why not the same as obs space?
        self.action_space = spaces.Box(low=JOINTS_MIN, high=JOINTS_MAX, shape=(len(ACTIONS),), dtype=np.float32)
        self.mujoco_actuators_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    
    def _make_env_task(self, task_name):
        # build the env according to the task type
        # table leg moving task
        if task_name == 'TableLegAssembleTask':
            xml_path = MUJOCO_DIR / 'so101_table_leg_assemble.xml'
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TableLegAssembleTask(observation_height = self.observation_height,
                                    observation_width = self.observation_width)
        else:
            raise NotImplementedError()
        
        # finally build the env itself
        env = control.Environment(physics, task, float("inf"), control_timestep=DT, n_sub_steps=None, flat_observation=False)
        
        return env
    
    def get_joint_range(self):
        actuators_ids = [self._env.physics.model.name2id(jid,'joint') for jid in self.mujoco_actuators_names]
        return self._env.physics.model.jnt_range[actuators_ids]
    
    def _format_raw_obs(self, raw_obs):
        # for pixels only return the images
        if self.obs_type == "pixels" or "pixels_agent_pos":
            obs = {
                "pixels": {
                    "top_cam": raw_obs["images"]["top_cam"].copy(),
                    "wrist_cam": raw_obs["images"]["wrist_cam"].copy(),
                },
            }
        # for pixels and agent state only return the images and agent pos
        if self.obs_type == "pixels_agent_pos":
            obs['agent_pos'] = raw_obs["qpos"]
        
        # other cases not implemented
        elif self.obs_type == "pixels_agent_pos_state":
            raise NotImplementedError()
        
        return obs
    
    def step(self, action):
        # make sure the action is one time step
        assert action.ndim == 1
        # make step TODO are we sure about the order of returns
        _, reward, _, raw_obs = self._env.step(action)  
        
        terminated = is_success = reward == self._env.task.max_reward
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
        image = self._env.physics.render(height=height, width=width, camera_id="iso_cam")
        return image
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # set seed
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)
        
        # reset env
        raw_obs = self._env.reset()
        observation = self._format_raw_obs(raw_obs.observation)
        info = {"is_success": False}
        
        return observation, info
    
    def close(self):
        pass