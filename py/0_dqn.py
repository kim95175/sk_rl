import tensorflow as tf
import gym
import random
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
from collections import deque
from tqdm import tqdm, trange, notebook

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size) # replay buffer 

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

# Neural Network Model 
class Model(tf.keras.Model):
    def __init__(self, num_actions, units=[32, 32]):
        super().__init__(name='basic_dqn')
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
        #print(q_values)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0]


env = gym.make('CartPole-v0')
act_size = env.action_space.n
units=[32, 32]
epsilon=1.0
min_epsilon=.01
epsilon_decay=0.995
train_nums=5000
buffer_size=200
start_learning=20
batch_size=8
gamma=0.95
target_update_iter=400

replay_buffer = ReplayBuffer(buffer_size)


# initialize action-value function (network) Q 
#       and target action-value function (network) Q' 
model = Model(act_size, units) 
target_model = Model(act_size, units)
target_model.set_weights(model.get_weights()) # initialize target network weight 
# network optimizer
opt = ko.Adam(learning_rate=.0015)  
model.compile(optimizer=opt, loss='mse')

# evaluate 
epi_rewards = []
n_episodes = 10
for i in range(n_episodes):
    obs, done, epi_reward = env.reset(), False, 0.0 # Using [None] to extend its dimension (4,) -> (1, 4)
    while not done :
        action = model.action_value(obs[None])
        #print(action)
        obs, reward, done, _ = env.step(action)
        epi_reward += reward
    print("{} episode reward : {}".format(i, epi_reward))
    epi_rewards.append(epi_reward)

mean_reward = np.mean(epi_rewards)
std_reward = np.std(epi_rewards)

print(f"mean_reward : {mean_reward:.2f} +/- {std_reward:.2f}")

print(" ==== train with dqn ==== ")
# initialize the initial observation of the agent
model = Model(act_size, units) 
target_model = Model(act_size, units)
target_model.set_weights(model.get_weights()) # initialize target network weight 
# network optimizer
opt = ko.Adam(learning_rate=.0015)  
model.compile(optimizer=opt, loss='mse')

obs = env.reset()
epi_rewards = []
epi_reward = 0.0
epi = 0 # number of episode taken
last_10_game_reward = deque(maxlen=10) # store last 10 episode rewards
for t in notebook.tqdm(range(1, train_nums+1), desc='train with DQN'):
    # epsilon update
    if epsilon > min_epsilon:
        epsilon = max(epsilon * epsilon_decay, min_epsilon)

    ##### step 1 #####
    #       Select action using episolon-greedy    

    # select action that maximize Q value f
    best_action = model.action_value(obs[None])  # input the obs to the network model // obs : (4, ) -> obs[None] : (1, 4)
    # e-greedy
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = best_action   # with prob. epsilon, select a random action
    

    ##### step 2 #####
    #       Take step and store transition to replay buffer
    
    next_obs, reward, done, _ = env.step(action)    # Excute action in the env to return s'(next state), r, done
    epi_reward += reward
    replay_buffer.store(obs, action, reward/100.0, next_obs, done)  # store transition in replay memory
    
    obs = next_obs  # s <- s'

    ##### step 3 #####
    #       Train network (perform gradient descent)

    if t > start_learning:  
        # sample random minibatch of transitions (s, a, r, s', done) from replay buffer
        s_batch, a_batch, r_batch, ns_batch, done_batch = replay_buffer.sample(batch_size)
        #print(s_batch, a_batch, r_batch, ns_batch, done_batch)
        # calculate target values r + gamma * maxQ(s', a') using target Q network
        target_q = r_batch + gamma * np.amax(target_model.predict(ns_batch), axis=1) * (1 - done_batch) # (batch_size, )
        # get action values from Q network
        q_values = model.predict(s_batch)   # (batch_size, 2)
        
        # update q_value
        for i, val in enumerate(a_batch): # (batch_size, )
            q_values[i][val] = target_q[i]

        # perform a gradient descent on Q network
        # Ths loss measures the mean squared error between prediction and target
        print(s_batch, q_values)    
        model.train_on_batch(s_batch, q_values)

    ##### step 4 #####
    #       Update target network     
    if t % target_update_iter == 0:
        target_model.set_weights(model.get_weights()) # assign the current network parameters to target network
            
    # if episode ends (done)
    if done:
        epi += 1 # num of episode 
        last_10_game_reward.append(epi_reward)
        avg_reward = np.mean(last_10_game_reward) 
        obs = env.reset() # environment reset 
        if epi % 10 == 0:
            print("[Episode {:>5}] epi reward: {:>6.2f}  --eps : {:>4.2f} --steps : {:>5}".format(epi, epi_rewards[-1], epsilon, t))
        epi_rewards.append(avg_reward)
        epi_reward = 0.0
            

epi_rewards = []
# After training    
for i in range(n_episodes):
    obs, done, epi_reward = env.reset(), False, 0.0 # Using [None] to extend its dimension (4,) -> (1, 4)
    while not done :
        action = model.action_value(obs[None])
        obs, reward, done, _ = env.step(action)
        epi_reward += reward
    print("{} episode reward : {}".format(i, epi_reward))
    epi_rewards.append(epi_reward)

mean_reward = np.mean(epi_rewards)
std_reward = np.std(epi_rewards)

print(f"mean_reward : {mean_reward:.2f} +/- {std_reward:.2f}")
