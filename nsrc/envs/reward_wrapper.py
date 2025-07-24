from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from nsrc.config import Config

cfg = Config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IntrinsicRewardWrapper(gym.RewardWrapper):
    def __init__(self, obs_env, act_env, model, beta=1.0, frame_stack_size=4, norm=False, act_dim=3):
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
        self.obs_env = obs_env
        self.act_env = act_env
        self.norm_func = RunningEMANormalizer() if norm else None

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
        action = F.one_hot(torch.tensor(action).unsqueeze(0), num_classes=3).float()


        obs_tensor = torch.tensor(stacked_obs.astype(np.float32) / 255.0)
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C*stack, H, W)

        next_obs_tensor = torch.tensor(stacked_next_obs.astype(np.float32) / 255.0)
        next_obs_tensor = next_obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C*stack, H, W)

        intrinsic = self.model.compute_intrinsic_reward(obs=obs_tensor, next_obs=next_obs_tensor,
                                                        action=action).item()

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

class RunningEMANormalizer:
    def __init__(self, alpha=0.99, epsilon=1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.ema_mean = 0.0
        self.ema_mean_sq = 0.0
        self.counter = 1

    def update(self, x):
        self.counter += 1
        self.ema_mean = self.alpha * self.ema_mean + (1 - self.alpha) * x
        self.ema_mean_sq = self.alpha * self.ema_mean_sq + (1 - self.alpha) * (x ** 2)

    def std(self):
        mean = self.ema_mean / (1 - self.alpha ** self.counter)
        mean_sq = self.ema_mean_sq / (1 - self.alpha ** self.counter)
        var = max(mean_sq - mean ** 2, 0.0)
        return (var + self.epsilon) ** 0.5

    def normalize(self, x):
        self.update(x)
        return x / self.std()

