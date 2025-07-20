import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from sb3_contrib import RecurrentPPO

model = RecurrentPPO.load("ppo_recurrent")

env = gym.make("MiniGrid-Empty-Random-6x6-v0", render_mode='human')
env = RGBImgPartialObsWrapper(env)  # Convert to RGB image observation
env = ImgObsWrapper(env)
model.set_env(env)
vec_env = model.get_env()

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones
    vec_env.render("human")