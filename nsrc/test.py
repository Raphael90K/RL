import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from sb3_contrib import RecurrentPPO

from torch.distributions import OneHotCategoricalStraightThrough

from nsrc.intrinsic.rnd_model import RNDConvModel

model = RecurrentPPO.load("ppo_recurrent_rnd")
# ----------------- RND SETUP --------------------
obs_shape = (3, 7, 7)
rnd_model = RNDConvModel(obs_shape)
obs_buffer = []  # Buffer to store observations for RND updates


env = gym.make("MiniGrid-Empty-Random-6x6-v0", render_mode='human', max_steps=50)
env = OneHotCategoricalStraightThrough(env)
env = ImgObsWrapper(env)
env.action_space = gym.spaces.discrete.Discrete(3)

print(env.action_space)

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