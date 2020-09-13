import grid2op
from grid2op.PlotGrid import PlotMatplot 
from grid2op.Reward import L2RPNReward
from tqdm import tqdm, notebook # for easy progress bar
from grid_agent import GridAgent

import numpy as np
import collections
import random

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
from collections import deque
from tqdm import tqdm, notebook  # 학습 과정을 더 깔끔하게 보여주는 library 입니다.

env = grid2op.make("rte_case5_example", test=True,  reward_class=L2RPNReward)

# Grid 환경에 맞춰 Conveter를 추가한 Agent 사용.
agent = GridAgent(env=env)

buffer_size=5000
learning_rate=0.01
epsilon=1.0
epsilon_decay=0.99
min_epsilon=0.01
gamma=0.98
batch_size=16
target_update_iter=400
train_nums=5000
max_iter=200

# Neural Network Model 
class Model(tf.keras.Model):
    def __init__(self, num_actions, units=[32, 32]):
        super().__init__()
        self.fc1 = kl.Dense(units[0], activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(units[1], activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(num_actions, name='q_values')

    # forward propagation
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    # return best action that maximize action-value (Q) from network
    # a* = argmax_a' Q(s, a')
    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0]

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size) 

    # store transition of each step in replay buffer
    def store(self, s, a, r, next_s, d):
        experience = (s, a, r, d, next_s)
        self.buffer.append(experience)
        self.count += 1

    # Sample random minibatch of transtion
    def sample(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch, a_batch, r_batch, d_batch, s2_batch = map(np.array, list(zip(*batch)))
        return s_batch, a_batch, r_batch, s2_batch, d_batch
    
    def clear(self):
        self.buffer.clear()
        self.count = 0


replay_buffer = ReplayBuffer(buffer_size)

network = Model(agent.num_actions)
target_network = Model(agent.num_actions)
target_network.set_weights(network.get_weights()) # initialize target network weight 
opt = ko.Adam(learning_rate=.0015, clipvalue=10.0)  # do gradient clip
network.compile(optimizer=opt, loss='mse')

obs = env.reset()
epi_duration = 0
epi = 0 # number of episode taken
epsilon=1.0


for t in notebook.tqdm(range(1, train_nums+1), desc='train with DQN'):
    # epsilon update
    if epsilon > min_epsilon:
        epsilon = max(epsilon * epsilon_decay, min_epsilon)

    #######################  step 1  ####################### 
    ####        Select action using episolon-greedy      ### 
    ########################################################   

    gym_obs = agent.convert_obs(obs)
    # select action that maximize Q value f
    best_action = network.action_value(gym_obs[None])  # input the obs to the network model // obs : (4, ) -> obs[None] : (1, 4)
    
    # e-greedy
    if np.random.rand() < epsilon:
        #action = np.random.randint(agent.num_actions)
        gym_act = 0
    else:
        gym_act = best_action   # with prob. epsilon, select a random action
    
    #######################  step 2  ####################### 
    #### Take step and store transition to replay buffer ### 
    ########################################################
    
    grid_act = agent.id_to_act(gym_act)
    next_obs, reward, done, _ = env.step(grid_act)    # Excute action in the env to return s'(next state), r, done
    gym_next_obs = agent.convert_obs(next_obs)
    replay_buffer.store(gym_obs, gym_act, reward, gym_next_obs, done)
    
    epi_duration += 1
    
    #######################  step 3  ####################### 
    ####     Train network (perform gradient descent)    ### 
    ########################################################
    
    # target value 계산
    # np.amax -> list 에서 가장 큰 값 반환
    s_batch, a_batch, r_batch, ns_batch, done_batch = replay_buffer.sample(batch_size)
    target_q = r_batch + gamma * np.amax(target_network.predict(ns_batch), axis=1) * (1- done_batch)  
    q_values = network.predict(s_batch) 
    for i, action in enumerate(a_batch):
        q_values[i][action] = target_q[i]
        
    network.train_on_batch(s_batch, q_values)
    
    #######################  step 3  ####################### 
    ####             Update target network               ### 
    ########################################################
      
    if t % target_update_iter == 0:
        target_network.set_weights(network.get_weights()) # assign the current network parameters to target network
 
    obs = next_obs  # s <- s'
    # if episode ends (done)
    if (epi_duration >= max_iter) or done:
        epi += 1 # num of episode 
        if epi % 20 == 0:
            print("[Episode {:>5}] epi duration: {:>6.2f}  --eps : {:>4.2f} --steps : {:>5}".format(epi, epi_duration, epsilon, t))
        obs, done, epi_duration = env.reset(), False, 0.0  # Environmnet reset


max_steps = env.chronics_handler.max_timestep()

# test before train
for i in range(5):
    obs, done, ep_reward = env.reset(), False, 0
    actions = []
    for t in notebook.tqdm(range(max_steps), desc=msg):
        converted_obs = agent.convert_obs(obs)
        action = network.action_value(converted_obs[None])
        actions.append(action)
        converted_act = agent.id_to_act(action)
        obs, reward, done, _ = env.step(converted_act)
        ep_reward += reward
        if done:
            break
    print(actions[:10])