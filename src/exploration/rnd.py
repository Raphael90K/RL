import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_exploration import CuriosityModule

class RNDModel(nn.Module):
    def __init__(self, input_shape, output_size=32):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        c, h, w = input_shape

        temp_encoder = nn.Sequential(
            nn.Conv2d(c, 8, 3, stride=2), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feature_dim = temp_encoder(dummy).shape[1]

        self.target = nn.Sequential(
            nn.Conv2d(c, 8, 3, stride=2), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(feature_dim, output_size)
        ).to(self.device)

        self.predictor = nn.Sequential(
            nn.Conv2d(c, 8, 3, stride=2), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(64, output_size)
        ).to(self.device)

        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)

    def compute_intrinsic_reward(self, obs):
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            target_feat = self.target(obs_tensor)
        pred_feat = self.predictor(obs_tensor)
        reward = F.mse_loss(pred_feat, target_feat, reduction='mean').item() * 100.0
        return reward

    def update(self, obs):
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            target_feat = self.target(obs_tensor)
        pred_feat = self.predictor(obs_tensor)
        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RNDWrapper(CuriosityModule):
    def __init__(self, obs_shape, *_):
        self.model = RNDModel(obs_shape)

    def compute_intrinsic_reward(self, obs, next_obs=None, action=None):
        return self.model.compute_intrinsic_reward(obs)

    def update(self, obs, next_obs=None, action=None):
        self.model.update(obs)