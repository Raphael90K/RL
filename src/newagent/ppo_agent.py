import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from gymnasium.vector import AsyncVectorEnv
from minigrid.wrappers import RGBImgPartialObsWrapper


# --- Running Mean Std
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count


# --- GAE
def compute_gae(rewards, values, dones, gamma, lam):
    advantages = np.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
    returns = advantages + values[:-1]
    return advantages, returns


# --- Models
class ActorCriticSeparate(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(64 * 3 * 3, 128), nn.ReLU())
        self.policy = nn.Linear(128, action_dim)
        self.value_ext = nn.Linear(128, 1)
        self.value_int = nn.Linear(128, 1)

    def forward(self, x):
        x = x / 255.0
        conv = self.conv(x)
        flat = torch.flatten(conv, 1)
        features = self.fc(flat)
        logits = self.policy(features)
        value_ext = self.value_ext(features)
        value_int = self.value_int(features)
        return logits, value_ext, value_int


class RNDNet(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 52 * 52, 256), nn.ReLU()
        )
        self.target = nn.Sequential(nn.Linear(256, feature_dim))
        self.predictor = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, feature_dim))
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x / 255.0
        conv = self.conv(x)
        target_out = self.target(conv)
        pred_out = self.predictor(conv)
        return target_out, pred_out


# --- Environments
def make_env():
    def thunk():
        env = gym.make("MiniGrid-FourRooms-v0", max_steps=150)
        env = RGBImgPartialObsWrapper(env)
        return env
    return thunk


# --- Training
def main():
    num_envs = 8
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)], shared_memory=False)
    obs_shape = (3, 56, 56)
    action_dim = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ppo_model = ActorCriticSeparate(action_dim).to(device)
    rnd_model = RNDNet().to(device)

    optimizer_ppo = optim.Adam(ppo_model.parameters(), lr=1e-4)
    optimizer_rnd = optim.Adam(rnd_model.predictor.parameters(), lr=1e-5, weight_decay=1e-5)

    int_rms = RunningMeanStd()

    gamma = 0.99
    lam = 0.95
    intrinsic_coeff = 1.0
    ppo_epochs = 4
    clip_eps = 0.2
    total_updates = 1000

    episodes_per_env = [0] * num_envs
    obs, _ = envs.reset()
    obs = torch.tensor(obs["image"], dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    episode_rewards = deque(maxlen=100)
    update = 0

    while update < total_updates:
        obs_buffer, actions_buffer, logprobs_buffer = [], [], []
        ext_rewards_buffer, int_rewards_buffer = [], []
        ext_values_buffer, int_values_buffer = [], []
        done_buffer, images_rnd_buffer = [], []

        while not all(count >= 10 for count in episodes_per_env):
            logits, ext_value, int_value = ppo_model(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            next_obs_img = torch.tensor(next_obs["image"], dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                target, pred = rnd_model(next_obs_img)
                intrinsic_reward = ((target - pred) ** 2).mean(dim=1).cpu().numpy()

            obs_buffer.append(obs)
            actions_buffer.append(action)
            logprobs_buffer.append(log_prob)
            ext_rewards_buffer.append(torch.tensor(reward, dtype=torch.float32, device=device))
            int_rewards_buffer.append(torch.tensor(intrinsic_reward, dtype=torch.float32, device=device))
            ext_values_buffer.append(ext_value)
            int_values_buffer.append(int_value)
            done_buffer.append(torch.tensor(done, dtype=torch.float32, device=device))
            images_rnd_buffer.append(next_obs_img)

            for i, info in enumerate(infos):
                if "episode" in info:
                    episodes_per_env[i] += 1
                    episode_rewards.append(info["episode"]["r"])

            obs = next_obs_img

        obs_tensor = torch.cat(obs_buffer)
        actions_tensor = torch.cat(actions_buffer)
        logprobs_tensor = torch.cat(logprobs_buffer)
        ext_rewards = torch.stack(ext_rewards_buffer).cpu().numpy()
        int_rewards = torch.stack(int_rewards_buffer).cpu().numpy()
        ext_values = torch.stack(ext_values_buffer).squeeze(-1).detach().cpu().numpy()
        int_values = torch.stack(int_values_buffer).squeeze(-1).detach().cpu().numpy()
        dones = torch.stack(done_buffer).cpu().numpy()

        int_rms.update(int_rewards)
        int_rewards = (int_rewards - int_rms.mean) / (np.sqrt(int_rms.var) + 1e-8)
        int_rewards = np.maximum(int_rewards, 0)

        ext_values = np.concatenate([ext_values, np.zeros((1, num_envs))], axis=0)
        int_values = np.concatenate([int_values, np.zeros((1, num_envs))], axis=0)

        adv_ext, ret_ext = compute_gae(ext_rewards, ext_values, dones, gamma, lam)
        adv_int, ret_int = compute_gae(int_rewards, int_values, np.zeros_like(dones), gamma, lam)

        advantages = adv_ext + intrinsic_coeff * adv_int
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = obs_tensor.reshape(-1, *obs_shape)
        actions_batch = actions_tensor.flatten()
        old_logprobs_batch = logprobs_tensor.flatten()

        advantages_batch = torch.tensor(advantages.flatten(), dtype=torch.float32, device=device)
        ret_ext_batch = torch.tensor(ret_ext.flatten(), dtype=torch.float32, device=device)
        ret_int_batch = torch.tensor(ret_int.flatten(), dtype=torch.float32, device=device)

        batch_size = obs_batch.shape[0]
        mini_batch_size = 256
        indices = np.arange(batch_size)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, mini_batch_size):
                print(f"PPO Batch Size: {batch_size}, Mini-Batches of {mini_batch_size}")
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                logits, val_ext, val_int = ppo_model(obs_batch[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_batch[mb_idx])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_logprobs_batch[mb_idx].detach()).exp()
                surr1 = ratio * advantages_batch[mb_idx]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_batch[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(val_ext.squeeze(), ret_ext_batch[mb_idx]) + \
                             F.mse_loss(val_int.squeeze(), ret_int_batch[mb_idx])

                loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

                optimizer_ppo.zero_grad()
                loss.backward()
                optimizer_ppo.step()

        rnd_loss = 0
        for img in images_rnd_buffer:
            target, pred = rnd_model(img)
            rnd_loss += F.mse_loss(pred, target.detach())
        rnd_loss /= len(images_rnd_buffer)
        optimizer_rnd.zero_grad()
        rnd_loss.backward()
        optimizer_rnd.step()

        update += 1
        episodes_per_env = [0] * num_envs

        print(f"Update {update}")
        print(f"Mean extrinsic reward: {np.mean(episode_rewards) if episode_rewards else 0:.4f}")
        print(f"RND Loss: {rnd_loss.item():.4f}")

        if update % 100 == 0:
            torch.save(ppo_model.state_dict(), f"../../models/{update}_ppo.pth")

    # --- Plotting
    plt.plot(episode_rewards)
    plt.title("Extrinsic Reward per Episode")
    plt.show()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
