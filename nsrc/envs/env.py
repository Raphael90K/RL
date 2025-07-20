import gymnasium as gym
import numpy as np

class Env:
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)  # Set action space to Discrete(3) for the environment

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)