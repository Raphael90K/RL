import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from nsrc.envs.observation_wrapper import SaveObsWrapper


def make_env(id = 'MiniGrid-FourRooms-v0'):
    env = gym.make(id, render_mode=None, max_steps=64)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = SaveObsWrapper(env)
    return env