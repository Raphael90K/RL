from collections import deque
import gymnasium as gym
import numpy as np
import torch

from nsrc.config import Config

cfg = Config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IntrinsicRewardWrapper(gym.RewardWrapper):
    def __init__(self, obs_env, act_env, model, beta=1.0, frame_stack_size=4, norm=False):
        super().__init__(obs_env)
        self.model = model
        self.beta = beta
        self.obs_buffer = model.obs_buffer
        self.next_obs_buffer = model.next_obs_buffer if hasattr(model, "next_obs_buffer") else None
        self.act_buffer = model.act_buffer if hasattr(model, "act_buffer") else None
        self.stack_size = frame_stack_size
        self.frames = deque(maxlen=frame_stack_size)
        self.intrinsic_rewards = []
        self.extrinsic_rewards = []
        self.visited_positions = set()  # Set to track visited positions for each environment
        self.obs_env = obs_env
        self.act_env = act_env
        self.norm_func = RunningMeanStdScalar() if norm else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if hasattr(self.model, "resets"):
            self.model.resets.append(len(self.obs_buffer))  # Track the number of resets
            self.model.reset_hidden_states()
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return obs, info

    def reward(self, reward):
        stacked_obs = np.concatenate(list(self.frames), axis=2)  # (H, W, C*stack)

        self.frames.append(self.obs_env.next_obs)
        stacked_next_obs = np.concatenate(list(self.frames), axis=2)  # (H, W, C*stack)

        action = self.act_env.last_action

        obs_tensor = torch.tensor(stacked_obs.astype(np.float32) / 255.0)
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C*stack, H, W)

        next_obs_tensor = torch.tensor(stacked_next_obs.astype(np.float32) / 255.0)
        next_obs_tensor = next_obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C*stack, H, W)

        intrinsic = self.model.compute_intrinsic_reward(obs=obs_tensor, next_obs=next_obs_tensor,
                                                        action=action).item()

        self.track_position()
        self.obs_buffer.append(stacked_obs.copy())
        if self.next_obs_buffer is not None:
            self.next_obs_buffer.append(stacked_next_obs.copy())
        if self.act_buffer is not None:
            self.act_buffer.append(action)

        self.extrinsic_rewards.append(reward)
        self.intrinsic_rewards.append(self.beta * intrinsic)

        if self.norm_func:
            intrinsic = self.norm_func.normalize(intrinsic)

        return reward + self.beta * intrinsic

    def reset_reward_buffers(self):
        self.intrinsic_rewards = []
        self.extrinsic_rewards = []

    def track_position(self):
        self.visited_positions.add(tuple(self.env.unwrapped.agent_pos))


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
