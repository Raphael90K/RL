import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .base_exploration import CuriosityModule


class BYOLNetwork(nn.Module):
    def __init__(self, input_shape, feature_dim=64, hidden_dim=128, momentum=0.996):
        super().__init__()
        c, h, w = input_shape

        temp_encoder = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feature_input_dim = temp_encoder(dummy).shape[1]

        self.online_encoder = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(feature_input_dim, feature_dim)
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)

        self.online_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.momentum = momentum


    @torch.no_grad()
    def update_target(self):
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data = self.momentum * target_p.data + (1 - self.momentum) * online_p.data

    def forward(self, x):
        z = self.online_encoder(x)
        p = self.online_predictor(z)
        return p


class BYOLWrapper(CuriosityModule):
    def __init__(self, obs_shape, *_):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.byol_model = BYOLNetwork(obs_shape).to(self.device)

    def compute_intrinsic_reward(self, obs, next_obs, action=None):
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            target_feature = self.byol_model.target_encoder(next_obs_tensor)

        predicted_feature = self.byol_model(obs_tensor)

        reward = F.mse_loss(F.normalize(predicted_feature, dim=-1),
                            F.normalize(target_feature, dim=-1), reduction='mean').item()

        return reward

    def update(self, obs, next_obs, action=None):
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        target_feature = self.byol_model.target_encoder(next_obs_tensor)
        predicted_feature = self.byol_model(obs_tensor)

        loss = F.mse_loss(F.normalize(predicted_feature, dim=-1), F.normalize(target_feature.detach(), dim=-1))

        self.byol_model.optimizer.zero_grad()
        loss.backward()
        self.byol_model.optimizer.step()

        self.byol_model.update_target()
