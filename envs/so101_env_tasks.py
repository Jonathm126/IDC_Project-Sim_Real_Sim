import collections

import numpy as np
from dm_control.suite import base

from envs.so101_env_utils import START_ARM_POSE

class SO101Task(base.Task):
    def __init__(self, random=None, observation_height = 480, observation_width = 640):
        super().__init__(random=random)
        self.observation_height = observation_height
        self.observation_width = observation_width
        # note that this parameter is set extenrally!
        self.start_pose = None
        
        # names
        self.actuators_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    
    def initialize_episode(self, physics):
        super().initialize_episode(physics)
    
    def get_qpos(self, physics):
        qpos_raw = physics.data.qpos.copy()
        return qpos_raw[self.actuators_ids]
    
    @staticmethod
    def get_qvel(ids, physics):
        raise NotImplementedError
    
    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError
    
    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        # obs["qvel"] = self.get_qvel(self.actuators_ids, physics)
        # obs["env_state"] = self.get_env_state(physics) TODO temporarys
        obs["images"] = {}
        obs["images"]["top_cam"]   = physics.render(height=self.observation_height, width=self.observation_width, camera_id="top_cam")
        obs["images"]["wrist_cam"] = physics.render(height=self.observation_height, width=self.observation_width, camera_id="wrist_cam")
        obs["images"]["iso_cam"]   = physics.render(height=self.observation_height, width=self.observation_width, camera_id="iso_cam")
        return obs
        
    def get_reward(self, physics):
        raise NotImplementedError

class TableLegAssembleTask(SO101Task):
    def __init__(self, random = None, observation_height = 480, observation_width = 640, start_pose = None):
        super().__init__(random, observation_height, observation_width)
        self.max_reward = 1
    
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            # if the start pose has belfen set externally:s
            start_pose = self.start_pose if self.start_pose is not None else START_ARM_POSE
            
            # set arm pose
            self.actuators_ids = [physics.model.name2id(jid,'joint') for jid in self.actuators_names]
            physics.named.data.qpos[self.actuators_ids] = start_pose
            np.copyto(physics.data.ctrl, start_pose)            
            
            # set peg and table pose
            rng = self.random
            physics.named.data.qpos['table_leg_joint'][:3] = [rng.uniform(0, 0.15), rng.uniform(-0.25, 0.25), 0.05]
            physics.named.data.qpos['table_top_joint'][:3] = [rng.uniform(0, 0.15), rng.uniform(-0.25, 0.25), 0.05]
            
            # set sites to clear
            self.sites = ['leg_tip', 'table_hole_1', 'table_hole_2', 'table_hole_3', 'table_hole_4']
            for site in self.sites:
                physics.named.model.site_rgba[site, 3] = 0.0
        
        # reset reward history
        self.history = {
            'gripper_grips'   : False,
            'peg_in_air'      : False,
            'peg_contact_hole': False,
        }
        super().initialize_episode(physics)
    
    def get_reward(self, physics):
        '''Rewards:
        1 - gripper grips peg
        2 - peg is lifted
        3 - peg is in contact with one of the holes
        4 - peg is left in the hole
        '''
        # site_id = physics.model.name2id("leg_tip", "site")
        # pos = physics.data.site_xpos[site_id]  # shape (3,)
        # quat = physics.data.site_xmat[site_id]  # shape (4,)
        
        # return reward based on state
        all_contact_body_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            id_body_1 = physics.model.geom_bodyid[id_geom_1]
            id_body_2 = physics.model.geom_bodyid[id_geom_2]
            name_body_1 = physics.model.id2name(id_body_1, "body")
            name_body_2 = physics.model.id2name(id_body_2, "body")
            contact_pair = (name_body_1, name_body_2)
            all_contact_body_pairs.append(contact_pair)
        
        # list of things the peg contacts
        peg_contacts = set()
        for a, b in all_contact_body_pairs:
            if a == "table_leg":
                peg_contacts.add(b)
            elif b == "table_leg":
                peg_contacts.add(a)

        gripper_bodies = {"gripper", "moving_jaw_so101_v1"}
        
        # is gripper gripping the peg?
        gripper_grips = len(peg_contacts & gripper_bodies) > 0
        # is peg only in contact with gripper?
        peg_in_air = gripper_grips and peg_contacts.issubset(gripper_bodies)
        
        #TODO
        peg_contact_hole = False
        
        # gripper grips the peg
        reward = 0
        if gripper_grips:
            reward = 1 / 4
            self.history['gripper_grips'] = True
        # gripper lifts the peg
        if peg_in_air:
            reward = 2 / 4
            self.history['peg_in_air'] = True
        # peg in hole and in gripper
        if peg_contact_hole and self.history['peg_in_air']:
            reward = 3 / 4
            self.history['peg_contact_hole'] = True
        # peg in hole, it was in the air and the gripper no longer grips
        if peg_contact_hole and self.history['peg_in_air'] and not gripper_grips:
            reward = 4 / 4
        
        return reward
