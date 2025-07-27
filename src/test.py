import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import Config
from src.envs.action_wrapper import SaveActionWrapper

if __name__ == "__main__":
    cfg = Config()
    cfg.set_seed()
    model = RecurrentPPO.load("../models/PLAIN_MiniGrid-MultiRoom-N2-S4-v0/PLAIN_checkpoint_200000_steps.zip",
                              device='cuda',
                              seed =cfg.seed,)

    env = gym.make(cfg.env_name, render_mode='human', max_steps=cfg.max_steps)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = SaveActionWrapper(env, cfg.allowed_actions)
    env.action_space = gym.spaces.discrete.Discrete(cfg.action_dim)
    vec_env = DummyVecEnv([lambda: env])  # Wrap the environment in a DummyVecEnv for vectorized operations
    vec_env.seed(cfg.seed)

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
        obs, rewards, dones, infos = vec_env.step(action)  # (H, W, C) -> (C, H, W)  # Convert to numpy array and change shape to (C, H, W)
        episode_starts = dones
