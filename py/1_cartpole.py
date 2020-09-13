import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
from collections import deque
from tqdm import tqdm, notebook  # 학습 과정을 더 깔끔하게 보여주는 library 입니다.

# CartPole-v0 라는 gym 환경을 만들어 env에 저장합니다.
env = gym.make('CartPole-v0')

# Box(4,) 는 4개의 요소로 구성된 벡터를 뜻합니다.
print("Observation space:", env.observation_space)
print("Observation space Max:", env.observation_space.high)
print("Observation space Min:", env.observation_space.low)
# Discrete(2) 는 개별적인 두 개의 action이 있음을 뜻합니다.
print("Action space:", env.action_space)
print("Action space num: ", env.action_space.n)

# reset은 매 episode가 시작할 때마다 호출해야 합니다.
obs = env.reset()

# random한 action을 뽑아 환경에 적용합니다..
action = env.action_space.sample()
print("Sampled action: {}\n".format(action))
obs, reward, done, info = env.step(action)

# info는 현재의 경우 비어있는 dict지만 debugging과 관련된 정보를 포함할 수 있습니다.
# reward는 scalar 값 입니다.
print("obs : {}\nreward : {}\ndone : {}\ninfo : {}\n".format(obs, reward, done, info))

# 한 episode 에 대한 testing
obs, done, ep_reward = env.reset(), False, 0

# 대부분의 gym 환경은 다음과 같은 흐름으로 진행됩니다.
while True: 
    action = env.action_space.sample() # action 선택
    obs, reward, done, info = env.step(action)  # 환경에 action 적용
    ep_reward += reward
    if done:  # episode 종료 여부 체크
        break
        
env.close()  
# Cartpole에서 reward = episode 동안 지속된 step 을 뜻합니다.
print("episode reward : ", ep_reward) 

# Neural Network Model 
class Model(tf.keras.Model):
    def __init__(self, num_actions, units=[32, 32]):
        super().__init__(name='basic_dqn')
        self.fc1 = kl.Dense(units[0], activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(units[1], activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(num_actions, name='q_values')

    # forward propagation
    def call(self, inputs):
        #inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
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

units=[32, 32]         # network의 구조. [32, 32]로 설정시 두개의 hidden layer에 32개의 node로 구성된 network가 생성
epsilon=1.0            # epsilon의 초기 값
min_epsilon=.01        # epsilon의 최솟값
epsilon_decay=0.995    # 매 step마다 epsilon이 줄어드는 비율 
train_nums=5000        # train이 진행되는 총 step
gamma=0.95             # discount factor

buffer_size=200        # Replay buffer의 size
batch_size=8           # Repaly buffer로 부터 가져오는 transition minbatch의 크기

target_update_iter=200 # Target network가 update 되는 주기 (step 기준)

network = Model(2)
print("network id ", id(network))
# network의 optimizer 와 loss 함수를 정의해 줍니다.
opt = ko.Adam(learning_rate=.0015, clipvalue=10.0)  # do gradient clip
network.compile(optimizer=opt, loss='mse')

# test before train
epi_rewards = []
n_episodes = 10
for i in range(n_episodes):
    obs, done, epi_reward = env.reset(), False, 0.0 
    while not done:
        action = network.action_value(obs[None]) # Using [None] to extend its dimension (4,) -> (1, 4)
        next_obs, reward, done, _ = env.step(action)
        epi_reward += reward
        obs = next_obs
    
    print("{} episode reward : {}".format(i, epi_reward))
    epi_rewards.append(epi_reward)

mean_reward = np.mean(epi_rewards)
std_reward = np.std(epi_rewards)

print(f"mean_reward : {mean_reward:.2f} +/- {std_reward:.2f}")

act_size=2
replay_buffer = ReplayBuffer(buffer_size)
network = Model(act_size, units)
target_network = Model(act_size, units)
print("network id {} target network id {}".format(id(network), id(target_network)))
target_network.set_weights(network.get_weights()) # initialize target network weight 

opt = ko.Adam(learning_rate=.0015, clipvalue=10.0)  # do gradient clip
network.compile(optimizer=opt, loss='mse')

obs = env.reset()
epi_reward = 0.0
epi = 0 # number of episode taken
epsilon=1.0

for t in notebook.tqdm(range(1, train_nums+1), desc='train with DQN'):
    # epsilon update
    if epsilon > min_epsilon:
        epsilon = max(epsilon * epsilon_decay, min_epsilon)

    #######################  step 1  ####################### 
    ####        Select action using episolon-greedy      ### 
    ########################################################   

    # select action that maximize Q value f
    best_action = network.action_value(obs[None])  # input the obs to the network model // obs : (4, ) -> obs[None] : (1, 4)
    
    # e-greedy
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = best_action   # with prob. epsilon, select a random action
    
    #######################  step 2  ####################### 
    #### Take step and store transition to replay buffer ### 
    ########################################################
    
    next_obs, reward, done, _ = env.step(action)    # Excute action in the env to return s'(next state), r, done
    epi_reward += reward
    replay_buffer.store(obs, action, reward, next_obs, done)
    
    #######################  step 3  ####################### 
    ####     Train network (perform gradient descent)    ### 
    ########################################################
    
    # target value 계산
    # np.amax -> list 에서 가장 큰 값 반환
    s_batch, a_batch, r_batch, ns_batch, done_batch = replay_buffer.sample(batch_size)
    #print(s_batch, a_batch, r_batch, ns_batch, done_batch)
    target_q = r_batch + gamma * np.amax(target_network.predict(ns_batch), axis=1) * (1- done_batch)  
    q_values = network.predict(s_batch) 

    for i, action in enumerate(a_batch):
        q_values[i][action] = target_q[i]

    print(s_batch.shape, q_values.shape)    
    network.train_on_batch(s_batch, q_values)
    
    #######################  step 3  ####################### 
    ####             Update target network               ### 
    ########################################################
      
    if t % target_update_iter == 0:
        target_network.set_weights(network.get_weights()) # assign the current network parameters to target network
 
    obs = next_obs  # s <- s'
    # if episode ends (done)
    if done:
        epi += 1 # num of episode 
        if epi % 20 == 0:
            print("[Episode {:>5}] epi reward: {:>6.2f}  --eps : {:>4.2f} --steps : {:>5}".format(epi, epi_reward, epsilon, t))
        obs, done, epi_reward = env.reset(), False, 0.0  # Environmnet reset
            