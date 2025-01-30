import gymnasium as gym
from stable_baselines3 import A2C
from pickle_functions import *

env = gym.make('CartPole-v1', render_mode='none')

model = A2C("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=1000000)

model.save("model_cart_pole")

# See tensorboard: tensorboard --logdir ./a2c_cartpole_tensorboard/
