import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback


class ICMModel(nn.Module):
    def __init__(self, obs_shape, obs_buffer, next_obs_buffer, act_buffer, action_dim, feature_dim=256, beta=0.2):
        super().__init__()
        print(beta)
        c, h, w = obs_shape
        self.feature = nn.Sequential(
            nn.Conv2d(c, 16, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * (h // 8) * (w // 8), feature_dim),
        )
        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )
        # Inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

        self.action_dim = action_dim
        self.obs_buffer = obs_buffer
        self.next_obs_buffer = next_obs_buffer
        self.act_buffer = act_buffer
        self.beta = beta
        self.device = torch.device("cuda")

    def forward(self, obs, next_obs, action):
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        action = action.to(self.device)

        features_obs = self.feature(obs)
        features_next_obs = self.feature(next_obs)

        concat_input = torch.cat([features_obs, action], dim=1)
        predicted_next_features = self.forward_model(concat_input)

        # Inverse model predicts action logits
        inverse_input = torch.cat([features_obs, features_next_obs], dim=1).to(self.device)
        predicted_action_logits = self.inverse_model(inverse_input)

        return features_obs, features_next_obs, predicted_next_features, predicted_action_logits

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs, next_obs, action):
        _, features_next_obs, predicted_next_features, _ = self.forward(obs, next_obs, action)
        intrinsic_reward = 0.5 * (predicted_next_features - features_next_obs.detach()).pow(2).sum(dim=1)
        return intrinsic_reward.detach()

    def to_long_tensor(self, x):
        if isinstance(x, torch.Tensor):
            tensor = x.detach().clone().to(dtype=torch.long, device=self.device)
        else:
            tensor = torch.tensor(x, dtype=torch.long, device=self.device)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

class ICMUpdateCallback(BaseCallback):
    def __init__(self, icm_model, lr=1e-3,
                 verbose=0):
        super().__init__(verbose)
        self.icm_model = icm_model
        self.optimizer = optim.Adam(self.icm_model.parameters(), lr=lr)
        self.obs_buffer = icm_model.obs_buffer
        self.next_obs_buffer = icm_model.next_obs_buffer
        self.act_buffer = icm_model.act_buffer

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        device = next(self.icm_model.parameters()).device
        obs_batch = torch.tensor(np.stack(self.obs_buffer), dtype=torch.float32).permute(0, 3, 1, 2).to(device) / 255.0
        next_obs_batch = torch.tensor(np.stack(self.next_obs_buffer), dtype=torch.float32).permute(0, 3, 1, 2).to(
            device) / 255.0
        actions = torch.tensor(np.stack(self.act_buffer), dtype=torch.long).to(device)
        actions = actions.squeeze(1)

        features_obs, features_next_obs, predicted_next_features, predicted_action_logits = self.icm_model(obs_batch,
                                                                                                           next_obs_batch,
                                                                                                           actions)

        actions_tensor = actions.argmax(dim=1).long()

        forward_loss = 0.5 * (features_next_obs - predicted_next_features).pow(2).sum(dim=1).mean()
        inverse_loss = nn.functional.cross_entropy(predicted_action_logits, actions_tensor)

        total_loss = self.icm_model.beta * forward_loss + (1 - self.icm_model.beta) * inverse_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm_model.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.logger.record('icm/forward_loss', forward_loss.item())
        self.logger.record('icm/inverse_loss', inverse_loss.item())
        self.logger.record('icm/total_loss', total_loss.item())

        self.obs_buffer.clear()
        self.next_obs_buffer.clear()
        self.act_buffer.clear()
