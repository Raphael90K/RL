import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_exploration import CuriosityModule

class ICM(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.device = torch.device('cuda')
        self.num_actions = num_actions

        c, h, w = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
        ).to(self.device)
        dummy = self.encoder(torch.zeros(1, *input_shape).to(self.device))
        self.feature_dim = dummy.shape[1]

        self.inverse_model = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        ).to(self.device)

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + num_actions, 256), nn.ReLU(),
            nn.Linear(256, self.feature_dim)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def encode(self, x):
        x = x.float().to(self.device) / 255.0
        return self.encoder(x)

    def compute_intrinsic_reward(self, obs, next_obs, action):
        obs = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
        next_obs = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
        action_onehot = F.one_hot(torch.tensor([action]), num_classes=self.num_actions).float().to(self.device)

        phi_s = self.encode(obs)
        phi_s_next = self.encode(next_obs)
        input_forward = torch.cat([phi_s, action_onehot], dim=1)
        phi_s_next_pred = self.forward_model(input_forward)

        intrinsic_reward = 0.5 * F.mse_loss(phi_s_next_pred, phi_s_next.detach(), reduction="sum").item()
        return intrinsic_reward

    def update(self, obs, next_obs, action):
        obs = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
        next_obs = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
        action = torch.tensor([action]).to(self.device)

        phi_s = self.encode(obs)
        phi_s_next = self.encode(next_obs)
        action_onehot = F.one_hot(action, num_classes=self.num_actions).float().to(self.device)

        pred_action_logits = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=1))
        loss_inv = F.cross_entropy(pred_action_logits, action)

        input_forward = torch.cat([phi_s.detach(), action_onehot], dim=1)
        phi_s_next_pred = self.forward_model(input_forward)
        loss_fwd = F.mse_loss(phi_s_next_pred, phi_s_next.detach())

        loss = loss_inv + loss_fwd
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ICMWrapper(CuriosityModule):
    def __init__(self, obs_shape, n_actions):
        self.model = ICM(obs_shape, n_actions)

    def compute_intrinsic_reward(self, obs, next_obs, action):
        return self.model.compute_intrinsic_reward(obs, next_obs, action)

    def update(self, obs, next_obs, action):
        self.model.update(obs, next_obs, action)