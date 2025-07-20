from datetime import datetime

import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper
from sb3_contrib import RecurrentPPO
import torch
import torch.optim as optim
from intrinsic.rnd_model import RNDConvModel, RNDUpdateCallback
from envs.reward_wrapper import IntrinsicRewardWrapper
from nsrc.envs.action_wrapper import SaveActionWrapper
from nsrc.envs.observation_wrapper import SaveObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    # ----------------- RND SETUP --------------------
    obs_shape = (3, 56, 56)
    rnd_model = RNDConvModel()
    obs_buffer = []  # Buffer to store observations for RND updates

    # ----------------- ENV SETUP --------------------
    env = gym.make("MiniGrid-FourRooms-v0", render_mode=None, max_steps=64)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    act_env = SaveActionWrapper(env)
    obs_env = SaveObsWrapper(act_env)  # Save observations for RND
    env_reward = IntrinsicRewardWrapper(obs_env, act_env, rnd_model, beta=1.0, obs_buffer=obs_buffer)
    env = Monitor(env_reward)  # Monitor to track rewards and other metrics
    env.action_space = gym.spaces.discrete.Discrete(3) # Set action space to Discrete(3) for the environment

    # ----------------- PPO SETUP -----------------
    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=2,
        tensorboard_log="./ppo_rnd_tensorboard/",
        ent_coef=0.05,
        device="cuda",
        n_epochs=5,
        n_steps=512,
        batch_size=256,
        seed=42
    )

    # ----------------- TRAINING -----------------
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    callback = RNDUpdateCallback(rnd_model, obs_buffer, f'./ppo_rnd_tensorboard/RND{time}', env_reward, lr=1e-5)
    model.learn(1_000_000, callback=callback, tb_log_name=f"PPO{time}")
    model.save("ppo_recurrent_rnd")

