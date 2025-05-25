import gymnasium as gym
from minigrid.wrappers import OneHotPartialObsWrapper, ImgObsWrapper


def make_four_rooms_env(render_mode=None, max_steps=500):
    raw_env = gym.make("MiniGrid-Empty-5x5-v0", render_mode=render_mode, max_steps=max_steps)  # Ensure the environment state is cloned
    env = OneHotPartialObsWrapper(raw_env)
    env = ImgObsWrapper(env)
    return env
