import grid2op
from grid2op.Agent import DoNothingAgent, AgentWithConverter
from grid2op.Converter import GymActionSpace, GymObservationSpace, IdToAct
from dqn_agent import DQNAgent, Model
from collections import deque
import numpy as np
from tqdm import notebook

import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko


class GridAgent(DQNAgent):
    def __init__(self, env, units=[32, 32], buffer_size=5000, learning_rate=0.01, init_epsilon=1.,
                epsilon_decay=0.99, min_epsilon=0.01, gamma=0.98, batch_size=16, 
                target_update_iter=300, train_nums=5000, start_learning=64, max_iter = 200):
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
        '''
        print("=======================alll action ==================")
        for i in range(ACT_SIZE):
            print("action number = ", i)
            print(self.converter.convert_act(i))
        print("buffer_size : ", buffer_size)
        print("env : ", self.env)
        '''
        self.units = units
        DQNAgent.__init__(self, self.env, obs = init_obs, num_actions = ACT_SIZE,
                buffer_size=buffer_size, learning_rate=learning_rate, init_epsilon=init_epsilon, 
                epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, gamma=gamma, 
                batch_size=batch_size, target_update_iter=target_update_iter, 
                train_nums=train_nums, start_learning=start_learning)
        #print(self.gym_action_space)
        self.max_iter = max_iter

    # grid observation -> gym observation
    def convert_obs(self, obs):
        gym_obs = self.gym_obs_space.to_gym(obs)  
        obs = []
        for key, val in gym_obs.items():
            if key in self.obs_list:
                obs.extend(val)
        return np.array(obs)
    
    # grid action id (int) -> grid action
    def id_to_act(self, encoded_act):
        grid2op_action = self.converter.convert_act(encoded_act)  # action for grid2op (abstract) 
        return grid2op_action

        
    def grid_evaluation(self, action_type = 'network', epi = 0):
        done = False
        obs, done, ep_reward, steps = self.env.reset(), False, 0, 0
        msg = 'SmartGrid Episode {}'.format(epi)
        for t in notebook.tqdm(range(1, 2017), desc=msg):
            converted_obs = self.convert_obs(obs)
            #print("convt_obs : ",converted_obs)
            if action_type == "network":
                action = self.model.action_value(converted_obs[None])
            elif action_type == "random":
                action= np.random.randint(self.num_actions)
            elif action_type == "do_nothing":
                action = 0
            #print("act : ",action)
            converted_act = self.id_to_act(action)
            #print("convt_act : ",converted_act)
            obs, reward, done, _ = self.env.step(converted_act)
            ep_reward += reward
            steps += 1
            if done:
                break
        self.env.close()
        return steps, ep_reward
