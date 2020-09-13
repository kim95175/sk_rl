import tensorflow as tf
import gym
import random
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import grid2op
from collections import deque
from tqdm import  trange, notebook

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size)

    # store transition 
    def store(self, s, a, r, next_s, d):
        experience = (s, a, r, d, next_s)
        self.buffer.append(experience)
        self.count += 1
    
    def size(self):
        return self.count
    
    # 
    def sample(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        s_batch, a_batch, r_batch, d_batch, s2_batch = map(np.array, list(zip(*batch)))
        #print(s_batch, a_batch, r_batch, d_batch, s2_batch)
        return s_batch, a_batch, r_batch, s2_batch, d_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

# Neural Network Model Defined at Here.
class Model(tf.keras.Model):
    def __init__(self, num_actions, units=[32, 32]):
        super().__init__(name='basic_dqn')
        # you can try different kernel initializer
        self.fc1 = kl.Dense(units[0], activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(units[1], activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(num_actions, name='q_values')

    # forward propagation
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    # a* = argmax_a' Q(s, a')
    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action if best_action.shape[0] > 1 else best_action[0]

class DQNAgent:  # Deep Q-Network
    def __init__(self, env, obs,
                units=[32, 32], num_actions=None, buffer_size=200, 
                learning_rate=.0015, init_epsilon=1.0, 
                epsilon_decay=0.995, min_epsilon=.01, 
                gamma=.95, batch_size=8, 
                target_update_iter=200, 
                train_nums=5000, start_learning=20,
                experience_replay = True, target_network = True):
        if num_actions == None:
            act_size = env.action_space.n 
        else:
            act_size = num_actions
        self.model = Model(act_size, units)
        self.target_model = Model(act_size, units)
        self.target_model.set_weights(self.model.get_weights())
        #print("nn, target nn id: ", id(self.model), id(self.target_model))
        # gradient clip
        opt = ko.Adam(learning_rate=learning_rate, clipvalue=10.0)  # do gradient clip
        self.model.compile(optimizer=opt, loss='mse')

        # parameters
        self.env = env                              # gym environment
        self.lr = learning_rate                     # learning step
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon                 # e-greedy when exploring
        self.epsilon_decay = epsilon_decay          # epsilon decay rate
        self.min_epsilon = min_epsilon              # minimum epsilon
        self.gamma = gamma                          # discount rate
        self.batch_size = batch_size                # batch_size
        self.target_update_iter = target_update_iter    # target network update period
        self.train_nums = train_nums                # total training steps
        self.buffer_size = buffer_size              # replay buffer size
        self.start_learning = start_learning        # step to begin learning(no update before that step)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.experience_replay = experience_replay
        self.target_network = target_network
        
    def train(self):
        if self.experience_replay == False:
            rewards = self.train_without_replay()
        elif self.target_network == False:
            rewards = self.train_without_target()
        else:
            rewards = self.train_dqn()
        
        return rewards
    
    def train_dqn(self):
        print(" ==== train with ddqn ==== ")
        # initialize the initial observation of the agent
        obs = self.env.reset()
        epi_rewards = []
        epi_reward = 0.0
        epi = 0
        last_10_game_reward = deque(maxlen=10)
        for t in notebook.tqdm(range(1, self.train_nums+1), desc='train with DDQN'):
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            best_action = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, _ = self.env.step(action)    # take the action in the env to return s', r, done
            epi_reward += reward
            self.replay_buffer.store(obs, action, reward/50.0, next_obs, done)  # store that transition into replay butter
            if t > self.start_learning:  # start learning
                self.train_network()
            if t % self.target_update_iter == 0:
                self.target_model.set_weights(self.model.get_weights()) # assign the current network parameters to target network
            if done:
                epi += 1
                last_10_game_reward.append(epi_reward)
                avg_reward = np.mean(last_10_game_reward)
                obs = self.env.reset()
                epi_rewards.append(avg_reward)
                if epi % 10 == 0:
                   print("[Episode {:>5}] epi reward: {:>6.2f}  --eps : {:>4.2f} --steps : {:>5}".format(epi, epi_rewards[-1], self.epsilon, t))
                epi_reward = 0.0
            else:
                obs = next_obs
        
        return epi_rewards

    
    def train_without_target(self):
        # initialize the initial observation of the agent
        obs = self.env.reset()
        epi_rewards = []
        epi_reward = 0.0
        epi = 0
        last_10_game_reward = deque(maxlen=10)
        for t in notebook.tqdm(range(1, self.train_nums+1), desc='train without target network'):
            # obs : (4, ) -> obs[None] : (1, 4)
            best_action = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, _ = self.env.step(action)    # take the action in the env to return s', r, done
            epi_reward += reward
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            self.replay_buffer.store(obs, action, reward/50.0, next_obs, done)  # store that transition into replay butter
            if t > self.start_learning:  # start learning
                s_batch, a_batch, r_batch, ns_batch, done_batch = self.replay_buffer.sample(self.batch_size)
                target_q = r_batch + self.gamma * np.amax(self.model.predict(ns_batch), axis=1) * (1 - done_batch)
                q_values = self.model.predict(s_batch)      
                for i, val in enumerate(a_batch): # (batch_size, )
                    q_values[i][val] = target_q[i]
                self.model.train_on_batch(s_batch, q_values)
            if done:
                epi += 1
                last_10_game_reward.append(epi_reward)
                avg_reward = np.mean(last_10_game_reward)
                obs = self.env.reset()
                if epi % 10 == 0:
                     print("[Episode {:>5}] epi reward: {:>6.2f}  --eps : {:>4.2f} --steps : {:>5}".format(epi, epi_rewards[-1], self.epsilon, t))
                epi_rewards.append(avg_reward)
                epi_reward = 0.0          
            else:
                obs = next_obs
        
        return epi_rewards
    
    def train_without_replay(self):
        obs = self.env.reset()
        epi_rewards = []
        epi_reward = 0.0
        epi = 0
        last_10_game_reward = deque(maxlen=10)
        for t in notebook.tqdm(range(1, self.train_nums+1), desc='train without experience replay'):
            # obs : (4, ) -> obs[None] : (1, 4)
            best_action = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, _ = self.env.step(action)    # take the action in the env to return s', r, done
            epi_reward += reward
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            target_q = reward + self.gamma * np.amax(self.target_model.predict(next_obs[None]), axis=1) * (1 - done)
            q_values = self.model.predict(obs[None])
            q_values[0][action] = target_q[0]
            losses = self.model.train_on_batch(obs[None], q_values)
            if t % self.target_update_iter == 0:
                self.target_model.set_weights(self.model.get_weights())
            if done:
                epi += 1
                last_10_game_reward.append(epi_reward)
                avg_reward = np.mean(last_10_game_reward)
                obs = self.env.reset()
                epi_rewards.append(avg_reward)                
                if epi % 10 == 0:
                    print("[Episode {:>5}] epi reward: {:>6.2f}  --eps : {:>4.2f} --steps : {:>5}".format(epi, epi_rewards[-1], self.epsilon, t))
                epi_reward = 0.0
            else:
                obs = next_obs
        
        return epi_rewards

    # implement double DQN
    def train_network(self):
        s_batch, a_batch, r_batch, ns_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        best_action_idxes = self.model.action_value(ns_batch)
        target_q = self.target_model.predict(ns_batch)
        target_q = r_batch + self.gamma * target_q[np.arange(target_q.shape[0]), best_action_idxes] * (1 - done_batch)
        #target_q = r_batch + self.gamma * np.amax(self.target_model.predict(ns_batch), axis=1) * (1 - done_batch) # (batch_size, )
        q_values = self.model.predict(s_batch)   # (batch_size, 2)
        
        for i, val in enumerate(a_batch): # (batch_size, )
            q_values[i][val] = target_q[i]

        losses = self.model.train_on_batch(s_batch, q_values)

        return losses

    # e-greedy
    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    def evaluate(self, n_episodes=10):
        epi_rewards = []
        for i in range(n_episodes):
            obs, done, epi_reward = self.env.reset(), False, 0.0 # Using [None] to extend its dimension (4,) -> (1, 4)
            while not done :
                action = self.model.action_value(obs[None])
                obs, reward, done, _ = self.env.step(action)
                epi_reward += reward
            print("{} episode reward : {}".format(i, epi_reward))
            epi_rewards.append(epi_reward)
        mean_reward = np.mean(epi_rewards)
        std_reward = np.std(epi_rewards)
        return mean_reward, std_reward

    

    

