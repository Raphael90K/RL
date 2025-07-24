import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from torch import optim, device

class RNDConvModel(nn.Module):
    def __init__(self, obs_buffer, output_dim=256):
        super().__init__()
        self.target = nn.Sequential(
            nn.Conv2d(12, 32, 5, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim)
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(12, 32, 5, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim)
        )

        for param in self.target.parameters():
            param.requires_grad = False

        for m in self.target.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

        for m in self.predictor.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

        self.obs_buffer = obs_buffer

    def forward(self, obs):
        device = next(self.parameters()).device
        obs = obs.to(device)
        with torch.no_grad():
            target_feature = self.target(obs)
        predictor_feature = self.predictor(obs)
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
        self.obs_buffer = rnd_model.obs_buffer


    def _on_step(self) -> bool:
        return True  # Pflicht-Implementierung, aber hier egal.

    def _on_rollout_end(self):
        if len(self.obs_buffer) == 0:
            print("no observations to update RND model")
            return
        obs_batch = torch.tensor(np.stack(self.obs_buffer), dtype=torch.float32)
        obs_batch = obs_batch.permute(0, 3, 1, 2)
        print(f'obs_batch shape: {obs_batch.shape}')
        pred, target = self.rnd_model(obs_batch)
        loss = (pred - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnd_model.predictor.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.obs_buffer.clear()
        self.logger.record('rnd/loss', loss.item())

