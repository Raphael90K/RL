import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from sb3_contrib import RecurrentPPO
import torch
import torch.optim as optim
from intrinsic.rnd_model import RNDConvModel, RNDUpdateCallback
from envs.reward_wrapper import IntrinsicRewardWrapper
from nsrc.envs.observation_wrapper import SaveObsWrapper
from stable_baselines3.common.monitor import Monitor

# ----------------- RND SETUP --------------------
obs_shape = (3, 56, 56)
rnd_model = RNDConvModel(obs_shape)
obs_buffer = []  # Buffer to store observations for RND updates

# ----------------- ENV SETUP --------------------
env = gym.make("MiniGrid-Empty-Random-6x6-v0", render_mode=None, max_steps=50)
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)
env = SaveObsWrapper(env)  # Save observations for RND
env = IntrinsicRewardWrapper(env, rnd_model, beta=10.0, obs_buffer=obs_buffer)
env = Monitor(env)  # Monitor to track rewards and other metrics
env.action_space = gym.spaces.discrete.Discrete(3) # Set action space to Discrete(3) for the environment

print(env.observation_space)

# ----------------- PPO SETUP -----------------
model = RecurrentPPO(
    "CnnLstmPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_rnd_tensorboard/",
    ent_coef=0.1,
    device="cuda",
)

# ----------------- TRAINING -----------------
callback = RNDUpdateCallback(rnd_model, obs_buffer, lr=1e-5)
model.learn(10_000, callback=callback, tb_log_name="PPO_RND")
model.save("ppo_recurrent_rnd")

