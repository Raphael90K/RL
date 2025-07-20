import gymnasium as gym
import numpy as np
import torch


class IntrinsicRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rnd_model, beta=1.0, obs_buffer=None):
        super().__init__(env)
        self.rnd_model = rnd_model
        self.beta = beta
        self.obs_buffer = obs_buffer

    def reward(self, reward):
        obs = self.env.last_obs
        obs = obs.astype(np.float32) / 255.0
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 7, 7)
        intrinsic = self.rnd_model.compute_intrinsic_reward(obs_tensor).item()
        self.obs_buffer.append(obs.copy())
        return reward + self.beta * intrinsic
