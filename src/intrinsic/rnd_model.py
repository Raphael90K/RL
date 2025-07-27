import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from torch import optim, device

class RNDConvModel(nn.Module):
    def __init__(self, obs_shape, next_obs_buffer, output_dim=256, feature_dim=256):
        super().__init__()
        c, h, w = obs_shape

        self.target = nn.Sequential(
            nn.Conv2d(c, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (h // 8) * (w // 8), feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim)
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(c, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (h // 8) * (w // 8), feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        for param in self.target.parameters():
            param.requires_grad = False

        for m in self.target.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

        for m in self.predictor.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

        self.next_obs_buffer = next_obs_buffer

    def forward(self, next_obs):
        device = next(self.parameters()).device
        next_obs = next_obs.to(device)
        with torch.no_grad():
            target_feature = self.target(next_obs)
        predictor_feature = self.predictor(next_obs)
        return predictor_feature, target_feature

    def compute_intrinsic_reward(self, obs, next_obs, action):

        pred, target = self.forward(next_obs)
        intrinsic_reward = (pred - target).pow(2).mean(dim=1)
        return intrinsic_reward.detach()


# ----------------- RND UPDATE CALLBACK -----------------
class RNDUpdateCallback(BaseCallback):
    def __init__(self, rnd_model, lr=1e-5, verbose=0):
        super().__init__(verbose)
        self.rnd_model = rnd_model
        self.optimizer = optim.Adam(rnd_model.predictor.parameters(), lr=lr)
        self.next_obs_buffer = rnd_model.next_obs_buffer


    def _on_step(self) -> bool:
        return True  # Pflicht-Implementierung, aber hier egal.

    def _on_rollout_end(self):
        if len(self.next_obs_buffer) == 0:
            print("no observations to update RND model")
            return
        obs_batch = torch.tensor(np.stack(self.next_obs_buffer), dtype=torch.float32) / 255.0
        obs_batch = obs_batch.permute(0, 3, 1, 2)
        print(f'obs_batch shape: {obs_batch.shape}')
        pred, target = self.rnd_model(obs_batch)
        loss = (pred - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnd_model.predictor.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.next_obs_buffer.clear()
        self.logger.record('rnd/loss', loss.item())

