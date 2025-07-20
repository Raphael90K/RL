import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback


class ICMModel(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=256):
        super().__init__()
        c, h, w = obs_shape
        self.feature = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (h // 4) * (w // 4), feature_dim),
            nn.ReLU()
        )
        # Forward model (predicts next feature from current feature + action)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

    def forward(self, obs, next_obs, action):
        phi = self.feature(obs)
        phi_next = self.feature(next_obs)
        inp = torch.cat([phi, action], dim=1)
        phi_next_pred = self.forward_model(inp)
        return phi_next, phi_next_pred


class ICMUpdateCallback(BaseCallback):
    def __init__(self, icm_model, obs_buffer, next_obs_buffer, action_buffer, log_dir, reward_wrapper, lr=1e-4, verbose=0):
        super().__init__(verbose)
        self.icm_model = icm_model
        self.optimizer = optim.Adam(self.icm_model.parameters(), lr=lr)
        self.obs_buffer = obs_buffer
        self.next_obs_buffer = next_obs_buffer
        self.action_buffer = action_buffer
        self.reward_wrapper = reward_wrapper
        self.writer = SummaryWriter(log_dir=log_dir)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        obs_batch = torch.tensor(np.stack(self.obs_buffer), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        next_obs_batch = torch.tensor(np.stack(self.next_obs_buffer), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        actions = torch.tensor(np.stack(self.action_buffer), dtype=torch.float32)
        phi_next, phi_next_pred = self.icm_model(obs_batch, next_obs_batch, actions)
        loss = (phi_next - phi_next_pred).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('icm/loss', loss.item(), self.num_timesteps)
        self.obs_buffer.clear()
        self.next_obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_wrapper.reset_reward_buffers()

    def _on_training_end(self):
        self.writer.close()
