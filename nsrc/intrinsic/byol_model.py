from zoneinfo import reset_tzpath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.callbacks import BaseCallback


class BYOLExploreModel(nn.Module):
    def __init__(self, obs_shape, action_dim, obs_buffer, next_obs_buffer, act_buffer,
                 ema_decay=0.99, feature_dim=256, device="cuda"):
        super().__init__()
        c, h, w = obs_shape
        self.device = torch.device(device)

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
        self.target_encoder = make_encoder()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.closed_rnn = nn.GRU(feature_dim + action_dim, feature_dim, batch_first=True)
        self.open_rnn = nn.GRU(action_dim, feature_dim, batch_first=True)
        for param in self.open_rnn.parameters():
            param.requires_grad = True

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.ema_decay = ema_decay
        self.obs_buffer = obs_buffer
        self.next_obs_buffer = next_obs_buffer
        self.act_buffer = act_buffer

        self.resets_flag = []
        self._prev_action = None  # zur Zwischenspeicherung von a_{t-1}

    def update_target(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data

    def compute_intrinsic_reward(self, obs, next_obs, action):
        with torch.no_grad():
            if self._prev_action is None:
                action_prev = torch.zeros_like(action)
            else:
                action_prev = self._prev_action

            obs= obs.to(self.device)
            next_obs = next_obs.to(self.device)
            action_prev = action_prev.to(self.device)
            action = action.to(self.device)

            omega_t = self.online_encoder(obs)
            omega_t = omega_t.unsqueeze(1)  # [B, 1, F]
            action_prev = action_prev.unsqueeze(1)  # [B, 1, A]

            closed_input = torch.cat([omega_t, action_prev], dim=-1)
            action = action.unsqueeze(1) # [B, 1, A]
            b_t, _ = self.closed_rnn(closed_input)
            b_open, _ = self.open_rnn(action, b_t.transpose(0, 1))
            b_open = b_open.squeeze(1)
            pred = self.predictor(b_open)
            target = self.target_encoder(next_obs).detach()
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)
            reward = 1 - F.cosine_similarity(pred, target, dim=-1)

            # aktionsspeicher aktualisieren
            self._prev_action = action.squeeze(1)
            return reward

    def reset_action_prev(self):
        self._prev_action = None


class BYOLExploreUpdateCallback(BaseCallback):
    def __init__(self, byol_model, lr=1e-4, log_dir=None, verbose=0):
        super().__init__(verbose)
        self.byol_model = byol_model
        self.optimizer = optim.Adam(self.byol_model.parameters(), lr=lr)

    def _on_step(self):
        dones = self.locals.get('dones', None)
        for done in dones:
            if done:
                self.byol_model.resets_flag.append(True)
            else:
                self.byol_model.resets_flag.append(False)
        return True

    def _on_rollout_end(self):
        device = self.byol_model.device
        obs = torch.tensor(np.stack(self.byol_model.obs_buffer), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        next_obs = torch.tensor(np.stack(self.byol_model.next_obs_buffer), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        actions = torch.tensor(np.stack(self.byol_model.act_buffer), dtype=torch.float32, device=device)
        resets = self.byol_model.resets_flag

        loss = self.train_on_sequence(obs, next_obs, actions, resets)

        self.byol_model.update_target()

        self.byol_model.obs_buffer.clear()
        self.byol_model.next_obs_buffer.clear()
        self.byol_model.act_buffer.clear()
        self.byol_model.resets_flag.clear()

        self.logger.record('byol_explore/loss', loss)

        return True


    def train_on_sequence(self, obs_batch, next_obs_batch, actions, resets_flag):
        online_encoder = self.byol_model.online_encoder
        target_encoder = self.byol_model.target_encoder
        closed_rnn = self.byol_model.closed_rnn
        open_rnn = self.byol_model.open_rnn
        predictor = self.byol_model.predictor

        losses = []
        hidden_closed = None
        for t in range(len(obs_batch) - 1):
            if resets_flag[t]:
                hidden_closed = None
                act_prev = torch.zeros_like(actions[t])  # Dummy action
            else:
                act_prev = actions[t - 1]

            obs_t = obs_batch[t].unsqueeze(0)
            next_obs_t = next_obs_batch[t].unsqueeze(0)
            act_prev = act_prev.unsqueeze(0)
            action = actions[t].unsqueeze(0)

            omega_t = online_encoder(obs_t).unsqueeze(1)
            closed_input = torch.cat([omega_t, act_prev], dim=-1)
            b_t, hidden_closed = closed_rnn(closed_input, hidden_closed)
            b_open, _ = open_rnn(action, b_t.transpose(0, 1))
            b_open = b_open.squeeze(1)

            pred = predictor(b_open)
            target = target_encoder(next_obs_t).detach()

            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)

            loss = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.byol_model.parameters(), 10.0)
        self.optimizer.step()

        return total_loss.item()
