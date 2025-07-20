import torch
import torch.nn as nn


class RNDConvModel(nn.Module):
    def __init__(self, obs_shape, output_dim=512):
        super().__init__()
        c, h, w = obs_shape
        self.target = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * ((h - 4) // 2) * ((w - 4) // 2), output_dim)
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * ((h - 4) // 2) * ((w - 4) // 2), output_dim)
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
