from collections import deque
import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt


class IntrinsicRewardWrapper(gym.RewardWrapper):
    def __init__(self, obs_env, act_env, model, obs_buffer, beta=1.0, frame_stack_size=4, norm=False):
        super().__init__(obs_env)
        self.model = model
        self.beta = beta
        self.obs_buffer = obs_buffer
        self.stack_size = frame_stack_size
        self.frames = deque(maxlen=frame_stack_size)
        self.intrinsic_rewards = []
        self.extrinsic_rewards = []
        self.obs_env = obs_env
        self.act_env = act_env
        self.norm_func = RunningMeanStdScalar() if norm else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return obs, info

    def reward(self, reward):
        next_obs = self.obs_env.next_obs
        last_obs = self.frames[-1] if self.frames else next_obs
        self.frames.append(next_obs)

        action = self.act_env.last_action

        stacked_obs = np.concatenate(list(self.frames), axis=2)  # (H, W, C*stack)

        obs_tensor = torch.tensor(stacked_obs.astype(np.float32) / 255.0)
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C*stack, H, W)

        intrinsic = self.model.compute_intrinsic_reward(obs_tensor, last_obs=last_obs, next_obs=next_obs,
                                                        action=action).item()
        self.obs_buffer.append(stacked_obs.copy())

        self.extrinsic_rewards.append(reward)
        self.intrinsic_rewards.append(self.beta * intrinsic)

        if self.norm_func:
            intrinsic = self.norm_func.normalize(intrinsic)

        return reward + self.beta * intrinsic

    def reset_reward_buffers(self):
        self.intrinsic_rewards = []
        self.extrinsic_rewards = []


class RunningMeanStdScalar:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        # x ist ein Skalar
        delta = x - self.mean
        self.count += 1
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var = ((self.count - 1) * self.var + delta * delta2) / self.count

    def std(self):
        return (self.var ** 0.5) + 1e-8  # vermeide Division durch 0

    def normalize(self, x):
        return (x - self.mean) / self.std()
