from stable_baselines3 import A2C
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')

model = A2C.load("model_cart_pole", env=env)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    # VecEnv resets automatically
    if done:
        obs = vec_env.reset()