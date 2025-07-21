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

        # CNN Encoder
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

        # RNN für zeitliche Konsistenz (wie im Paper)
        self.online_rnn = nn.GRU(feature_dim, feature_dim, batch_first=True)
        self.target_rnn = nn.GRU(feature_dim, feature_dim, batch_first=True)

        self.online_rnn.flatten_parameters()
        self.target_rnn.flatten_parameters()

        for param in self.target_rnn.parameters():
            param.requires_grad = False

        # Predictor Head (wie bei BYOL)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.ema_decay = ema_decay
        self.obs_buffer = obs_buffer
        self.next_obs_buffer = next_obs_buffer

        self.online_hidden = None
        self.target_hidden = None

        self.resets = []
        self.resets_flag = []
        self.online_hidden_training = None
        self.target_hidden_training = None

    def update_target(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data
        for online_param, target_param in zip(self.online_rnn.parameters(), self.target_rnn.parameters()):
            target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data

    def forward(self, obs, next_obs):
        # obs, next_obs: [B, C, H, W] -> [B, F]
        online_features = self.online_encoder(obs)
        target_features = self.target_encoder(next_obs).detach()

        # RNN erwartet [B, Seq, F] - hier "Seq" = 1
        online_features_seq = online_features.unsqueeze(1)
        target_features_seq = target_features.unsqueeze(1)

        online_features_rnn, self.online_hidden = self.online_rnn(online_features_seq, self.online_hidden)
        target_features_rnn, self.target_hidden = self.target_rnn(target_features_seq, self.target_hidden)

        online_features_rnn = online_features_rnn.squeeze(1)
        target_features_rnn = target_features_rnn.squeeze(1)

        pred_feature = self.predictor(online_features_rnn)

        return pred_feature, target_features_rnn

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs, next_obs, action):
        pred_feature, target_feature = self.forward(obs, next_obs)
        pred_feature = F.normalize(pred_feature, dim=-1)
        target_feature = F.normalize(target_feature, dim=-1)

        intrinsic_reward = 1 - F.cosine_similarity(pred_feature, target_feature, dim=-1)
        return intrinsic_reward  # pro Sample, nicht mean

    def reset_hidden_states(self):
        self.online_hidden = None
        self.target_hidden = None


class BYOLUpdateCallback(BaseCallback):
    def __init__(self, byol_model, lr=1e-4, log_dir=None, verbose=0):
        super().__init__(verbose)
        self.byol_model = byol_model
        self.optimizer = optim.Adam(self.byol_model.parameters(), lr=lr)
        self.obs_buffer = byol_model.obs_buffer
        self.next_obs_buffer = byol_model.next_obs_buffer
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        for done in dones:
            self.byol_model.resets_flag.append(done)
        return True

    def _on_rollout_end(self):
        device = self.byol_model.device
        obs_batch = (torch.tensor(np.stack(self.obs_buffer), dtype=torch.float32, device=device).
                     permute(0, 3, 1, 2) / 255.0)
        next_obs_batch = (torch.tensor(np.stack(self.next_obs_buffer), dtype=torch.float32, device=device).
                          permute(0, 3, 1,2) / 255.0)

        loss = self.train_on_buffer(obs_batch, next_obs_batch)
        self.byol_model.update_target()



        self.logger.record('byol/loss', loss.item())

        self.obs_buffer.clear()
        self.next_obs_buffer.clear()
        self.model.resets_flag = [False]

    def _on_training_end(self):
        if self.writer:
            self.writer.close()

    def train_on_buffer(self, obs_batch, next_obs_batch):

        (h1_temp, h2_temp) = self.byol_model.online_hidden, self.byol_model.target_hidden

        total_losses = []
        for i in range(len(self.byol_model.resets) - 1):
            start = self.byol_model.resets[i]
            end = self.byol_model.resets[i + 1] if i + 1 < len(self.byol_model.resets) else len(self.obs_buffer)
            episode_obs = obs_batch[start:end]
            episode_next_obs = next_obs_batch[start:end]
            resets_flags = self.byol_model.resets_flag[start:end]

            total_loss = self.optimize(episode_obs, episode_next_obs, resets_flags)
            total_losses.append(total_loss.item())

        self.byol_model.online_hidden = h1_temp
        self.byol_model.target_hidden = h2_temp
        self.byol_model.resets.clear()
        self.byol_model.resets_flag.clear()
        return np.array(total_losses).mean()

    def optimize(self, obs_seq, next_obs_seq, resets_flags):
        if len(obs_seq) == 0:
            return torch.tensor(0.0, device=self.byol_model.device)
        losses = []
        for obs, next_obs, reset in zip(obs_seq, next_obs_seq, resets_flags):

            self.byol_model.online_hidden = self.byol_model.online_hidden_training
            self.byol_model.target_hidden = self.byol_model.target_hidden_training

            pred, target = self.byol_model.forward(obs.unsqueeze(0), next_obs.unsqueeze(0))

            self.byol_model.online_hidden_training = self.byol_model.online_hidden
            self.byol_model.target_hidden_training = self.byol_model.target_hidden

            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)

            loss = 1 - (pred * target.detach()).sum(dim=-1).mean()
            losses.append(loss)

            if reset:
                self.byol_model.online_hidden_training = None
                self.byol_model.target_hidden_training = None

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.byol_model.parameters(), 10.0)
        self.optimizer.step()

        return total_loss
