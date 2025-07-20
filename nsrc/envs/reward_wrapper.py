from collections import deque
import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt


class IntrinsicRewardWrapper(gym.RewardWrapper):
    def __init__(self, obs_env, act_env, model, beta=1.0, obs_buffer=None, frame_stack_size=4):
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


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return obs, info

    def reward(self, reward):
        next_obs = self.obs_env.last_obs
        obs = self.frames[-1] if self.frames else next_obs
        self.frames.append(next_obs)

        action = self.act_env.last_action

        plt.imshow(obs)
        plt.show()

        plt.imshow(next_obs)
        plt.show()

        stacked_obs = np.concatenate(list(self.frames), axis=2)  # (H, W, C*stack)

        obs_tensor = torch.tensor(stacked_obs.astype(np.float32) / 255.0)
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C*stack, H, W)

        intrinsic = self.model.compute_intrinsic_reward(obs_tensor).item()
        self.obs_buffer.append(stacked_obs.copy())

        self.extrinsic_rewards.append(reward)
        self.intrinsic_rewards.append(self.beta * intrinsic)

        return reward + self.beta * intrinsic

    def reset_reward_buffers(self):
        self.intrinsic_rewards = []
        self.extrinsic_rewards = []
