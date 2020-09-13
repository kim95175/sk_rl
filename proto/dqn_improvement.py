import tensorflow as tf
import gym
import random
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import grid2op
from collections import deque

from tqdm import tqdm, trange, notebook
from per import SumTree

np.random.seed(1)
tf.random.set_seed(1)

# Neural Network Model Defined at Here.
class Model(tf.keras.Model):
    def __init__(self, num_actions, units=[32, 32], grid_model=False):
        super().__init__(name='basic_dqn')
        # you can try different kernel initializer
        self.fc1 = kl.Dense(units[0], activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(units[1], activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(num_actions, name='q_values')
        self.grid_model = grid_model

    # forward propagation
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    # a* = argmax_a' Q(s, a')
    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0], q_values[0]

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size)

    def add(self, s, a, r, next_s, d):
        experience = (s, a, r, d, next_s)
        self.buffer.append(experience)
        self.count += 1
    
    def size(self):
        return self.count
    
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


class DQNAgent:  # Deep Q-Network
    def __init__(self, env, obs, model = None, target_model = None,
                units=[32, 32], buffer_size=200, 
                learning_rate=.0015, init_epsilon=1.0, 
                epsilon_decay=0.995, min_epsilon=.01, 
                gamma=.95, batch_size=8, 
                target_update_iter=400, 
                train_nums=5000, start_learning=20, 
                experience_replay = True, target_network = True, **kwargs):
        
        act_size = env.action_space.n 
        if model == None:
            print("genrate model")
            self.model = Model(act_size, units)
            self.target_model = Model(act_size, units)
        else:
            print("get model from arg")
            self.model = model
            self.target_model = target_model
        print("nn, target nn id: ", id(self.model), id(self.target_model))
        self.update_target_model()  # to make sure the two models don't update simultaneously
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
        self.train_nums = train_nums                # total training ****steps****
        self.buffer_size = buffer_size              # replay buffer size
        self.start_learning = start_learning        # step to begin learning(no update before that step)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.experience_replay = experience_replay
        self.target_network = target_network

        print("-"*20)
        print("Experience Replay : {}\t Target Network : {}".format(self.experience_replay, self.target_network))
        print("-"*20)
        if kwargs is not None:
            self.double = kwargs['double']
            self.per = kwargs['per']
            print("Double : {}\t PER: {}".format(self.double, self.per))
            print("-"*20)
        else:
            self.double = False
            self.per = False

        if self.per :
            self.replay_buffer = SumTree(buffer_size)
            self.replay_period = 20
            self.alpha = 0.4
            self.beta = 0.4
            self.beta_increment_per_sample = 0.001
            self.margin = 0.01
            self.p1 = 1
            self.is_weight = np.power(self.buffer_size, - self.beta)
            self.abs_error_upper = 1

            self.b_obs = np.empty((self.batch_size,) + obs.shape)
            self.b_actions = np.empty((self.batch_size), dtype=np.int8)
            self.b_rewards = np.empty((self.batch_size), dtype=np.float32)
            self.b_dones = np.empty((self.batch_size), dtype=np.bool)
            self.b_next_states = np.empty((self.batch_size,) + obs.shape)
            self.b_next_idx = 0

    def train(self):
        if self.experience_replay == False:
            self.train_without_replay()
        elif self.target_network == False:
            self.train_without_target()
        else:
            self.dqn_train() 

    def dqn_train(self):
        # initialize the initial observation of the agent
        obs = self.env.reset()
        epi_rewards = []
        epi_reward = 0.0
        epi = 0
        last_10_game_reward = deque(maxlen=10)
        for t in notebook.tqdm(range(1, self.train_nums+1), desc='train with DQN'):
            # obs : (4, ) -> obs[None] : (1, 4)
            best_action, q_values = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, _ = self.env.step(action)    # take the action in the env to return s', r, done
            epi_reward += reward
            if self.per :
                if t==1 :
                    p =self.p1
                else:
                    p = np.max(self.replay_buffer.tree[-self.replay_buffer.capacity:])
                self.per_store_transition(p, obs, action, reward, next_obs, done)
                if t > self.start_learning:  # start learning
                    losses = self.per_train_step()
            else:
                self.store_transition(obs, action, reward, next_obs, done)  # store that transition into replay butter
                if t > self.start_learning:  # start learning
                    losses = self.train_step()
            if t % self.target_update_iter == 0:
                self.update_target_model()
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

    def train_without_target(self):
        # initialize the initial observation of the agent
        obs = self.env.reset()
        epi_rewards = []
        epi_reward = 0.0
        epi = 0
        last_10_game_reward = deque(maxlen=10)
        for t in notebook.tqdm(range(1, self.train_nums+1), desc='train without target network'):
            # obs : (4, ) -> obs[None] : (1, 4)
            best_action, q_values = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, _ = self.env.step(action)    # take the action in the env to return s', r, done
            epi_reward += reward
            self.store_transition(obs, action, reward, next_obs, done)  # store that transition into replay butter
            if t > self.start_learning:  # start learning
                s_batch, a_batch, r_batch, ns_batch, done_batch = self.replay_buffer.sample(self.batch_size)
                target_q = r_batch + self.gamma * np.amax(self.model.predict(ns_batch), axis=1) * (1 - done_batch)
                target_f = self.model.predict(s_batch)      
                for i, val in enumerate(a_batch): # (batch_size, )
                    target_f[i][val] = target_q[i]
                self.model.train_on_batch(s_batch, target_f)
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

    def train_without_replay(self):
        obs = self.env.reset()
        epi_rewards = []
        epi_reward = 0.0
        epi = 0
        last_10_game_reward = deque(maxlen=10)
        for t in notebook.tqdm(range(1, self.train_nums+1), desc='train without experience replay'):
            # obs : (4, ) -> obs[None] : (1, 4)
            best_action, q_values = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, _ = self.env.step(action)    # take the action in the env to return s', r, done
            epi_reward += reward
            if self.epsilon > self.min_epsilon:
                self.e_decay()
            target_q = reward + self.gamma * np.amax(self.get_target_value(next_obs[None]), axis=1) * (1 - done)
            target_f = self.model.predict(obs[None])
            target_f[0][action] = target_q[0]
            losses = self.model.train_on_batch(obs[None], target_f)
            if t % self.target_update_iter == 0:
                if self.target_network:
                    self.update_target_model()
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

    def train_step(self):
        s_batch, a_batch, r_batch, ns_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        if self.double:
            best_action_idx, _ = self.model.action_value(ns_batch)
            target_q = self.get_target_value(ns_batch)
            target_q = r_batch +  self.gamma * target_q[np.arange(target_q.shape[0]), best_action_idx] * (1 - done_batch)
            target_f = self.model.predict(s_batch)
        else:
            target_q = r_batch + self.gamma * np.amax(self.get_target_value(ns_batch), axis=1) * (1 - done_batch) # (batch_size, )
            target_f = self.model.predict(s_batch)   # (batch_size, 2)
        
        for i, val in enumerate(a_batch): # (batch_size, )
            target_f[i][val] = target_q[i]

        losses = self.model.train_on_batch(s_batch, target_f)

        return losses

    def evaluate(self, n_episodes=10):
        epi_rewards = []
        for i in range(n_episodes):
            obs, done, epi_reward = self.env.reset(), False, 0.0 # Using [None] to extend its dimension (4,) -> (1, 4)
            while not done :
                action, _ = self.model.action_value(obs[None])
                obs, reward, done, _ = self.env.step(action)
                epi_reward += reward
            print("{} episode reward : {}".format(i, epi_reward))
            epi_rewards.append(epi_reward)
        mean_reward = np.mean(epi_rewards)
        std_reward = np.std(epi_rewards)
        return mean_reward, std_reward

    # store transitions into replay butter
    def store_transition(self, obs, action, reward, next_state, done):
        self.replay_buffer.add(obs, action, reward, next_state, done)
        if self.epsilon > self.min_epsilon:
            self.e_decay()

    # e-greedy
    def get_action(self, best_action):
        #explore_probability = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * decay_step)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    # assign the current network parameters to target network
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

    def e_decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        
    def _per_loss(self, y_target, y_pred):
        return tf.reduce_mean(self.is_weight * tf.math.squared_difference(y_target, y_pred))

    def per_train_step(self):
        idxes, self.is_weight = self.sum_tree_sample(self.batch_size)
        best_action_idxes, _ = self.model.action_value(self.b_next_states)
        target_q = self.get_target_value(self.b_next_states)
        td_target = self.b_rewards + \
            self.gamma * target_q[np.arange(target_q.shape[0]), best_action_idxes] * (1- self.b_dones)
        predict_q = self.model.predict(self.b_obs)
        td_predict = predict_q[np.arange(predict_q.shape[0]), self.b_actions]
        abs_td_error = np.abs(td_target - td_predict) + self.margin
        clipped_error = np.where(abs_td_error < self.abs_error_upper, abs_td_error, self.abs_error_upper)
        ps = np.power(clipped_error, self.alpha)
        # priorities update
        for idx, p in zip(idxes, ps):
            self.replay_buffer.update(idx, p)

        for i, val in enumerate(self.b_actions):
            predict_q[i][val] = td_target[i]

        target_q = predict_q  # just to change a more explicit name
        losses = self.model.train_on_batch(self.b_obs, target_q)

        return losses
    
    def sum_tree_sample(self, k):
        idxes = []
        is_weights = np.empty((k, 1))
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        min_prob = np.min(self.replay_buffer.tree[-self.replay_buffer.capacity:]) / self.replay_buffer.total_p
        max_weight = np.power(self.buffer_size * min_prob, -self.beta)
        segment = self.replay_buffer.total_p / k
        for i in range(k):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, t = self.replay_buffer.get_leaf(s)
            idxes.append(idx)
            self.b_obs[i], self.b_actions[i], self.b_rewards[i], self.b_next_states[i], self.b_dones[i] = t
            # P(j)
            sampling_probabilities = p / self.replay_buffer.total_p     # where p = p ** self.alpha
            is_weights[i, 0] = np.power(self.buffer_size * sampling_probabilities, -self.beta) / max_weight
        
        return idxes, is_weights
    
    def per_store_transition(self, priority, obs, action, reward, next_state, done):
        transition = [obs, action, reward, next_state, done]
        self.replay_buffer.add(priority, transition)
        if self.epsilon > self.min_epsilon:
            self.e_decay()
    
