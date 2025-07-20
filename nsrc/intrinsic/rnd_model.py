import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class RNDConvModel(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.target = nn.Sequential(
            nn.Conv2d(12, 32, 5, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, output_dim)
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(12, 8, 5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 5, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16 * 35 * 35, output_dim)
        )
        for param in self.target.parameters():
            param.requires_grad = False

        for m in self.target.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

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
    def __init__(self, rnd_model, obs_buffer, log_dir, reward_wrapper, lr=1e-5, verbose=0):
        super().__init__(verbose)
        self.rnd_model = rnd_model
        self.optimizer = optim.Adam(rnd_model.predictor.parameters(), lr=lr)
        self.obs_buffer = obs_buffer
        self.reward_wrapper = reward_wrapper
        self.count = 50_000

        self.writer = SummaryWriter(log_dir=log_dir)

    def _on_step(self) -> bool:
        return True  # Pflicht-Implementierung, aber hier egal.

    def _on_rollout_end(self):

        if len(self.obs_buffer) == 0:
            print("no observations to update RND model")
            return
        extrinsic_mean = np.mean(self.reward_wrapper.extrinsic_rewards)
        intrinsic_mean = np.mean(self.reward_wrapper.intrinsic_rewards)

        self.writer.add_scalar('rnd/extrinsic_mean', extrinsic_mean, self.num_timesteps)
        self.writer.add_scalar('rnd/intrinsic_mean', intrinsic_mean, self.num_timesteps)

        self.reward_wrapper.reset_reward_buffers()

        obs_batch = torch.tensor(np.stack(self.obs_buffer), dtype=torch.float32)
        obs_batch = obs_batch.permute(0, 3, 1, 2)
        print(f'obs_batch shape: {obs_batch.shape}')
        pred, target = self.rnd_model(obs_batch)
        loss = (pred - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.obs_buffer.clear()
        self.writer.add_scalar('rnd/loss', loss.item(), self.num_timesteps)
        if self.num_timesteps > self.count:
            self.count += 50_000
            print(f'num_timesteps: {self.num_timesteps}')
            self.model.save('ppo_recurrent_rnd_rollout')

    def _on_training_end(self):
        self.writer.close()
