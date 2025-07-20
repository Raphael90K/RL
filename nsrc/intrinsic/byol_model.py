import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback


class BYOLModel(nn.Module):
    def __init__(self, obs_shape, feature_dim=256):
        super().__init__()
        c, h, w = obs_shape
        self.online_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (h // 4) * (w // 4), feature_dim),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.target_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (h // 4) * (w // 4), feature_dim),
            nn.ReLU()
        )
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.ema_decay = 0.99

    def update_target(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data

    def forward(self, obs, next_obs):
        online_feature = self.online_encoder(obs)
        pred_feature = self.predictor(online_feature)
        with torch.no_grad():
            target_feature = self.target_encoder(next_obs)
        return pred_feature, target_feature


class BYOLUpdateCallback(BaseCallback):
    def __init__(self, byol_model, obs_buffer, next_obs_buffer, log_dir, reward_wrapper, lr=1e-4, verbose=0):
        super().__init__(verbose)
        self.byol_model = byol_model
        self.optimizer = optim.Adam(self.byol_model.parameters(), lr=lr)
        self.obs_buffer = obs_buffer
        self.next_obs_buffer = next_obs_buffer
        self.reward_wrapper = reward_wrapper
        self.writer = SummaryWriter(log_dir=log_dir)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        obs_batch = torch.tensor(np.stack(self.obs_buffer), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        next_obs_batch = torch.tensor(np.stack(self.next_obs_buffer), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        pred, target = self.byol_model(obs_batch, next_obs_batch)
        loss = (pred - target.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.byol_model.update_target()

        self.writer.add_scalar('byol/loss', loss.item(), self.num_timesteps)
        self.obs_buffer.clear()
        self.next_obs_buffer.clear()
        self.reward_wrapper.reset_reward_buffers()

    def _on_training_end(self):
        self.writer.close()
