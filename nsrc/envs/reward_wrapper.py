import gymnasium as gym
import numpy as np
import torch


class IntrinsicRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rnd_model, beta=1.0):
        super().__init__(env)
        self.rnd_model = rnd_model
        self.beta = beta

    def reward(self, reward):
        obs = self.env.unwrapped.last_obs["image"]  # Achtung: das ist abhängig von Wrapper
        obs = obs.astype(np.float32) / 255.0
        obs = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        intrinsic = self.rnd_model.compute_intrinsic_reward(obs).item()
        return reward + self.beta * intrinsic
