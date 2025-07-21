import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from nsrc.envs.observation_wrapper import SaveObsWrapper


def make_env(id = 'MiniGrid-FourRooms-v0', max_steps=64, render_mode=None):
    env = gym.make(id, render_mode=render_mode, max_steps=max_steps)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env