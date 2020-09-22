import grid2op
from grid2op.Agent import DoNothingAgent, AgentWithConverter
from grid2op.Converter import GymActionSpace, GymObservationSpace, IdToAct
from collections import deque
import numpy as np
from tqdm import notebook

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko


class GridAgent():
    def __init__(self, env):
        # =============== Init Observation Space =================
        self.env = env
        self.gym_obs_space = GymObservationSpace(self.env)
        self.all_obs = []     
        obs = self.env.reset()
        gym_obs = self.gym_obs_space.to_gym(obs)
        for key in gym_obs.items():
            self.all_obs.append(key)
        self.obs_list = ["prod_p", "prod_v", "load_p", "load_q", "rho", "topo_vect"]     
        init_obs = self.convert_obs(obs)
        #print("obs_shape = ", init_obs.shape) # (39,)
        
        # =============== Init Action Space =================
        self.converter = grid2op.Converter.IdToAct(self.env.action_space)
        # action number 0 = DoNothing 
        self.converter.init_converter(set_line_status=False,   # 40 
                                    change_line_status=True,   # 8
                                    change_bus_vect=False,  # 59
                                    set_topo_vect=True, # 58
                                    ) 
        self.gym_action_space = GymActionSpace(action_space=self.converter)
        ACT_SIZE = len(self.converter.all_actions)
        #print("action space size= ", ACT_SIZE) # 68
        #print("gym_action_space = ", self.gym_action_space) # Dict(action:Discrete(68))
        gym_action = self.gym_action_space.sample()
        #print("sample_action = ", gym_action) # OrderedDict([('action', 34)])
        encoded_action = self.gym_action_space.from_gym(gym_action)  # 34
        self.num_actions = ACT_SIZE

    # grid observation -> gym observation
    def convert_obs(self, obs):
        gym_obs = self.gym_obs_space.to_gym(obs)  
        obs = []
        for key, val in gym_obs.items():
            if key in self.obs_list:
                obs.extend(val)
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        
        return obs
    
    # grid action id (int) -> grid action
    def id_to_act(self, encoded_act):
        grid2op_action = self.converter.convert_act(encoded_act)  # action for grid2op (abstract) 
        return grid2op_action
