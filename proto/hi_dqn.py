import tensorflow as tf
import gym
import random
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
from collections import deque
from tqdm import tqdm, trange, notebook
from matplotlib import pyplot as plt
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

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
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0]

class myDQNAgent:  # Deep Q-Network
    def __init__(self, env, obs,
                units=[64, 64], buffer_size=10000, 
                learning_rate=.0015, init_epsilon=1.0, 
                epsilon_decay=0.995, min_epsilon=.01, 
                gamma=.9, batch_size=32, 
                target_update_iter=100, 
                train_nums=5000, start_learning=20):
        
        print("===")
        act_size = env.action_space.n 

        # parameters
        self.env = env                              # gym environment
        self.lr = learning_rate                     # learning step
        self.init_epsilon = init_epsilon            # initial epsilon
        self.epsilon = init_epsilon                 # epsilon-greedy when exploring
        self.epsilon_decay = epsilon_decay          # epsilon decay rate
        self.min_epsilon = min_epsilon              # minimum epsilon
        self.gamma = gamma                          # discount rate
        self.batch_size = batch_size                # batch_size
        self.target_update_iter = target_update_iter    # target network update period
        self.train_nums = train_nums                # total training steps
        self.buffer_size = buffer_size              # replay buffer size
        self.start_learning = start_learning        # step to begin learning(no update before that step)
        
        # initialize replay memory to capacity N (buffuer_size)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # initialize action-value function (network) Q 
        #       and target action-value function (network) Q' 
        self.model = Model(act_size, units) 
        self.target_model = Model(act_size, units)
        self.target_model.set_weights(self.model.get_weights()) # initialize target network weight 
        # network optimizer
        opt = ko.Adam(learning_rate=learning_rate, clipvalue=10.0)  # do gradient clip
        self.model.compile(optimizer=opt, loss='mse')

        
    def train(self):
        print(" ==== train with dqn ==== ")
        # initialize the initial observation of the agent
        obs = self.env.reset()
        epi_rewards = []
        epi_reward = 0.0
        epi = 0 # number of episode taken
        last_10_game_reward = deque(maxlen=10) # store last 10 episode rewards
        #for t in notebook.tqdm(range(1, self.train_nums+1), desc='train with DQN'):
        for t in trange(1, self.train_nums+1):
            # epsilon update
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            
            # obs : (4, ) -> obs[None] : (1, 4)
            # select action that maximiㅋe Q value f
            obs = obs.flatten()
            best_action = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # with prob. epsilon, select a random action
            next_obs, reward, done, _ = self.env.step(action)    # Excute action in the env to return s'(next state), r, done
            epi_reward += reward
            
            # store transition in replay memory
            self.replay_buffer.store(obs, action, reward, next_obs.flatten(), done) 
            obs = next_obs  # s <- s'

            # train network (perform gradient descent)
            if t > self.start_learning:  # start learning
                self.train_network() 

            # assign the current network parameters to target network
            if t % self.target_update_iter == 0:
                self.target_model.set_weights(self.model.get_weights()) 
            
            # if episode ends (done)
            if done:
                epi += 1 # num of episode 
                last_10_game_reward.append(epi_reward)
                avg_reward = np.mean(last_10_game_reward) 
                obs = self.env.reset() # environment reset 
                if epi % 10 == 0:
                   print("[Episode {:>5}] epi reward: {:>6.2f}  --eps : {:>4.2f} --steps : {:>5}".format(epi, epi_rewards[-1], self.epsilon, t))
                epi_rewards.append(avg_reward)
                epi_reward = 0.0
            
        return epi_rewards


    def train_network(self):
        # sample random minibatch of transitions (s, a, r, s', done) from replay buffer
        s_batch, a_batch, r_batch, ns_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        # calculate target values r + gamma * maxQ(s', a') using target Q network
        target_q = r_batch + self.gamma * np.amax(self.target_model.predict(ns_batch), axis=1) * (1 - done_batch) # (batch_size, )
        # get action values from Q network
        q_values = self.model.predict(s_batch)   # (batch_size, 2)
        
        # update q_value
        for i, val in enumerate(a_batch): # (batch_size, )
            q_values[i][val] = target_q[i]

        # perform a gradient descent on Q network
        # Ths loss measures the mean squared error between prediction and target
        losses = self.model.train_on_batch(s_batch, q_values)

        return losses

    # e-greedy
    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    # evaluate network
    def evaluate(self, n_episodes=10):
        epi_rewards = []
        for i in range(n_episodes):
            obs, done, epi_reward = self.env.reset(), False, 0.0 # Using [None] to extend its dimension (4,) -> (1, 4)
            while not done :
                self.env.render()
                #print(obs.flatten().shape)
                obs = obs.flatten()
                action = self.model.action_value(obs[None])
                #print(action)
                obs, reward, done, _ = self.env.step(action)
                epi_reward += reward
            print("{} episode reward : {:.2f}".format(i, epi_reward))
            epi_rewards.append(epi_reward)
        plt.imshow(self.env.render(mode="rgb_array"))
        plt.show()
        mean_reward = np.mean(epi_rewards)
        std_reward = np.std(epi_rewards)
        return mean_reward, std_reward

    

    

