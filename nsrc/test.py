import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from torch.distributions import OneHotCategoricalStraightThrough

from nsrc.intrinsic.rnd_model import RNDConvModel

model = RecurrentPPO.load("ppo_recurrent_rnd_rollout")
# ----------------- RND SETUP --------------------
obs_shape = (24, 56, 56)
obs_buffer = []  # Buffer to store observations for RND updates
rnd_model = RNDConvModel(obs_buffer)


env = gym.make("MiniGrid-FourRooms-v0", render_mode='human', max_steps=50)
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)
env.action_space = gym.spaces.discrete.Discrete(3)
vec_env = DummyVecEnv([lambda: env])  # Wrap the environment in a DummyVecEnv for vectorized operations

print(env.action_space)

model.set_env(vec_env)

obs = vec_env.reset()

# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    print(action)
    obs, rewards, dones, infos = vec_env.step(action)
    episode_starts = dones
