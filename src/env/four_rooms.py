import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, OneHotPartialObsWrapper


def make_four_rooms_env(render_mode=None, max_episode_steps=500):
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode=render_mode, max_episode_steps=max_episode_steps)
    env = OneHotPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env
