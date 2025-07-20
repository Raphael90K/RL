import gymnasium as gym
import numpy as np

class FlattenRGBWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space["image"]
        flat_dim = int(np.prod(obs_space.shape))
        self.observation_space = gym.spaces.Box(0, 1, shape=(flat_dim,), dtype=np.float32)

    def observation(self, obs):
        img = obs["image"]
        return img.flatten().astype(np.float32) / 255.0
