import matplotlib.pyplot as plt
import numpy as np
import gym
from tqdm import  trange, notebook
import collections
from dqn import DQNAgent, Model 
from hi_dqn import myDQNAgent 
import highway_env

env = gym.make("highway-v0")
env.config["lanes_count"] = 2
env.config["observation"] =  {"type": "TimeToCollision","horizon": 10 }
env.config["vehicles_count"] = 5
env.config["duration"] = 20


# reset은 매 episode가 시작할 때마다 호출해야 합니다.
obs = env.reset()
# random한 action을 뽑습니다.
action = env.action_space.sample()
print("Sampled action:", action)
print("num_action ", env.action_space.n)
obs, reward, done, info = env.step(action)
# info는 현재의 경우 비어있는 dict지만 debugging과 관련된 정보를 포함할 수 있습니다.
# reward는 scalar 값 입니다.
print("obs : {}\nreward : {}\ndone : {}\ninfo : {}\n".format(obs, reward, done, info))

# 한 episode 에 대한 testing
obs, done, ep_reward = env.reset(), False, 0
for t in notebook.tqdm(range(200), desc='test '):
    action = env.action_space.sample()  
    obs, reward, done, info = env.step(action)
    ep_reward += reward
    if done:
        break
env.close()

print("episode reward : ", ep_reward)

UNITS = [32, 32]       # 학습하는 neural network의 layer 갯수
BUFFER_SIZE=200        # experience replay에 사용되는 buffer의 크기
LEARNING_RATE=0.015    # network가 한번에 학습하는 정도를 뜻한다.
EPSILON=1              # epsilon-greedy 에 사용 되는 초기 epsilon 값 
EPSILON_DECAY=0.995    # epsilon 값을 얼마나 빠르게 줄일 지를 나타낸다.
MIN_EPSILON=0.01       # 최소 epsilon 값
GAMMA=0.95             # Discounting factor
BATCH_SIZE=4           # 학습이 replay buffer로 부터 한번에 가져오는 experience의 양
TARGET_UPDATE_ITER=400 # target network가 update되는 주기
TRAIN_NUMS=5000        # 총 train 횟수 (step 단위)
START_LEARNING=20      # 학습이 시작되는 step

init_obs = env.reset()

agent = myDQNAgent(env=env, obs=init_obs, 
                batch_size=BATCH_SIZE, train_nums=5000)


# test before train
mean_reward, std_reward = agent.evaluate(n_episodes=5)
print(f"mean_reward : {mean_reward:.2f} +/- {std_reward:.2f}")
 
epi_rewards = agent.train()

# test after train            
mean_reward, std_reward = agent.evaluate(n_episodes=5)
print(f"mean_reward : {mean_reward:.2f} +/- {std_reward:.2f}")
