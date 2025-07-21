import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback


class BYOLModel(nn.Module):
    def __init__(self, obs_shape, obs_buffer, next_obs_buffer, ema_decay=0.99, feature_dim=256, device="cuda"):
        super().__init__()
        c, h, w = obs_shape
        self.device = torch.device(device)

        # Encoder Architektur
        def make_encoder():
            return nn.Sequential(
                nn.Conv2d(c, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * (h // 4) * (w // 4), feature_dim),
                nn.ReLU()
            )

        self.online_encoder = make_encoder()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.target_encoder = make_encoder()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.ema_decay = ema_decay
        self.obs_buffer = obs_buffer
        self.next_obs_buffer = next_obs_buffer


    def update_target(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data

    def forward(self, obs, next_obs):
        online_feature = self.online_encoder(obs)
        pred_feature = self.predictor(online_feature)
        with torch.no_grad():
            target_feature = self.target_encoder(next_obs)
        return pred_feature, target_feature

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs, next_obs, action):
        pred_feature, target_feature = self.forward(obs, next_obs)
        pred_feature = F.normalize(pred_feature, dim=-1)
        target_feature = F.normalize(target_feature, dim=-1)

        intrinsic_reward = 1 - F.cosine_similarity(pred_feature, target_feature, dim=-1)
        return intrinsic_reward.mean()


class BYOLUpdateCallback(BaseCallback):
    def __init__(self, byol_model, lr=1e-4, verbose=0):
        super().__init__(verbose)
        self.byol_model = byol_model
        self.optimizer = optim.Adam(self.byol_model.parameters(), lr=lr)
        self.obs_buffer = byol_model.obs_buffer
        self.next_obs_buffer = byol_model.next_obs_buffer

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        device = self.byol_model.device
        # Buffer in Tensor wandeln
        obs_batch = torch.tensor(np.stack(self.obs_buffer), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        next_obs_batch = torch.tensor(np.stack(self.next_obs_buffer), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0

        # Vorwärts
        pred, target = self.byol_model(obs_batch, next_obs_batch)

        # Optional: Cosine Loss für bessere Stabilität
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        loss = 2 - 2 * (pred * target.detach()).sum(dim=-1).mean()

        # Backward + Optimizer
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.byol_model.parameters(), 10.0)
        self.optimizer.step()

        self.byol_model.update_target()

        # Logging
        self.logger.record('byol/loss', loss.item())

        # Buffer leeren
        self.obs_buffer.clear()
        self.next_obs_buffer.clear()
