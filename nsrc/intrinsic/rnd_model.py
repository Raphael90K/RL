import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class RNDConvModel(nn.Module):
    def __init__(self, obs_shape, output_dim=256):
        super().__init__()
        c, h, w = obs_shape
        self.target = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, output_dim)
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, output_dim)
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        with torch.no_grad():
            target_feature = self.target(obs)
        predictor_feature = self.predictor(obs)
        return predictor_feature, target_feature

    def compute_intrinsic_reward(self, obs):
        pred, target = self.forward(obs)
        intrinsic_reward = (pred - target).pow(2).mean(dim=1)
        return intrinsic_reward.detach()


# ----------------- RND UPDATE CALLBACK -----------------
class RNDUpdateCallback(BaseCallback):
    def __init__(self, rnd_model, obs_buffer, lr=1e-5, verbose=0):
        super().__init__(verbose)
        self.rnd_model = rnd_model
        self.optimizer = optim.Adam(rnd_model.predictor.parameters(), lr=lr)
        self.obs_buffer = obs_buffer
        self.writer = SummaryWriter()

    def _on_step(self) -> bool:
        return True  # Pflicht-Implementierung, aber hier egal.

    def _on_rollout_end(self):
        if len(self.obs_buffer) == 0:
            print("no observations to update RND model")
            return
        obs_batch = torch.tensor(np.stack(self.obs_buffer), dtype=torch.float32)
        obs_batch = obs_batch.permute(0, 3, 1, 2)
        pred, target = self.rnd_model(obs_batch)
        loss = (pred - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.obs_buffer.clear()
        self.writer.add_scalar('rnd/loss', loss.item(), self.num_timesteps)

    def _on_training_end(self):
        self.writer.close()

