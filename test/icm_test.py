import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque


# --- CNN + LSTM Feature Extractor
class CNNLSTMActorCritic(nn.Module):
    def __init__(self, action_dim, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.Tanh(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.Tanh(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.Tanh()
        )
        self.ln = nn.LayerNorm(64 * 7 * 7)
        self.linear = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU()
        )
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward_features(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.ln(x)
        x = self.linear(x)
        return x

    def forward(self, x, hidden):
        x = self.forward_features(x)
        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        x = x.squeeze(1)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value, hidden, x

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, 128).to(device)
        c0 = torch.zeros(1, batch_size, 128).to(device)
        return (h0, c0)


# --- RND Networks
class RNDCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(56 * 56 * 16, 128), nn.ReLU()
        )
        self.target = nn.Sequential(nn.Linear(128, out_dim))
        self.predictor = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.conv(x)
        target_out = self.target(features)
        pred_out = self.predictor(features)
        return target_out, pred_out


# --- Hyperparameters
gamma = 0.99
clip_epsilon = 0.2
ppo_epochs = 4
intrinsic_reward_scale = 0.1
max_env_steps = 15_000_000
buffer_steps = 1500
max_steps_per_episode = 150

# --- Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = gym.make("MiniGrid-FourRooms-v0", render_mode=None, max_steps=max_steps_per_episode)
env = RGBImgPartialObsWrapper(env)
action_dim = 3

ppo_model = CNNLSTMActorCritic(action_dim).to(device)
rnd_model = RNDCNN().to(device)
optimizer_ppo = optim.Adam(ppo_model.parameters(), lr=1e-4)
optimizer_rnd = optim.Adam(rnd_model.predictor.parameters(), lr=1e-4)

width, height = env.unwrapped.width, env.unwrapped.height
position_heatmap = np.zeros((height, width))
intrinsic_rewards_buffer = deque(maxlen=10_000)

episode_rewards = []
extrinsic_episode_rewards = []
intrinsic_episode_rewards = []

# --- Buffer for PPO
rollout_obs = []
rollout_actions = []
rollout_logprobs = []
rollout_values = []
rollout_rewards = []
rollout_intrinsic = []
rollout_features = []

# --- Training Loop
global_env_steps = 0
episode = 0
while global_env_steps < max_env_steps:
    obs, _ = env.reset()
    hidden = ppo_model.init_hidden(batch_size=1, device=device)

    obs_img = torch.tensor(obs['image'], dtype=torch.float32, device=device) / 255.0
    obs_img = obs_img.unsqueeze(0)

    done = False
    positions = []

    extrinsic_sum = 0
    intrinsic_sum = 0

    while not done:
        positions.append(tuple(env.unwrapped.agent_pos))

        logits, value, hidden, lstm_out = ppo_model(obs_img, hidden)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        reward = 1.0 if reward > 0 else 0.0
        next_obs_img = torch.tensor(next_obs['image'], dtype=torch.float32, device=device) / 255.0
        next_obs_img = next_obs_img.unsqueeze(0)

        done = terminated or truncated

        target_feature, pred_feature = rnd_model(next_obs_img)
        intrinsic_reward = (target_feature - pred_feature).pow(2).mean().item()

        intrinsic_rewards_buffer.append(intrinsic_reward)
        intrinsic_array = np.array(intrinsic_rewards_buffer)
        global_mean = intrinsic_array.mean()
        global_std = intrinsic_array.std()

        normalized_intrinsic = (intrinsic_reward - global_mean) / (global_std + 1e-8)
        normalized_intrinsic = np.clip(normalized_intrinsic, 0, 5)

        rollout_obs.append(lstm_out.detach())
        rollout_actions.append(action)
        rollout_logprobs.append(log_prob)
        rollout_values.append(value)
        rollout_rewards.append(reward)
        rollout_intrinsic.append(normalized_intrinsic)
        rollout_features.append(next_obs_img)

        obs_img = next_obs_img
        extrinsic_sum += reward
        intrinsic_sum += normalized_intrinsic
        global_env_steps += 1

        for x, y in positions:
            position_heatmap[y, x] += 1

        if global_env_steps % buffer_steps == 0:
            # --- Process Rollout Buffer for PPO Update
            rewards = np.array(rollout_rewards)
            intrinsics = np.array(rollout_intrinsic)
            returns = []
            G = 0
            for ext_r, int_r in zip(reversed(rewards), reversed(intrinsics)):
                G = ext_r + intrinsic_reward_scale * int_r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)

            obs_tensor = torch.cat(rollout_obs)
            actions_tensor = torch.stack(rollout_actions)
            logprobs_tensor = torch.stack(rollout_logprobs)
            values_tensor = torch.cat(rollout_values).squeeze()

            advantage = returns - values_tensor.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for _ in range(ppo_epochs):
                logits = ppo_model.policy_head(obs_tensor)
                value = ppo_model.value_head(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs_new = dist.log_prob(actions_tensor)
                ratios = (log_probs_new - logprobs_tensor.detach()).exp()
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns - value.squeeze()).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy
                optimizer_ppo.zero_grad()
                loss.backward()
                optimizer_ppo.step()

            rnd_loss = 0
            for img in rollout_features:
                target, pred = rnd_model(img)
                rnd_loss += (target - pred).pow(2).mean()
            rnd_loss /= len(rollout_features)
            optimizer_rnd.zero_grad()
            rnd_loss.backward()
            optimizer_rnd.step()

            rollout_obs.clear()
            rollout_actions.clear()
            rollout_logprobs.clear()
            rollout_values.clear()
            rollout_rewards.clear()
            rollout_intrinsic.clear()
            rollout_features.clear()

    # --- Episode Done
    episode += 1
    extrinsic_episode_rewards.append(extrinsic_sum)
    intrinsic_episode_rewards.append(intrinsic_sum * intrinsic_reward_scale)
    episode_rewards.append(extrinsic_sum + intrinsic_sum * intrinsic_reward_scale)

    print(f"Episode {episode}: Extrinsic {extrinsic_sum:.2f}, Intrinsic {intrinsic_sum:.2f}, Total {extrinsic_sum + intrinsic_sum * intrinsic_reward_scale:.2f}")

    if episode % 50 == 0:
        print(f"Saving model at episode {episode}")
        torch.save(ppo_model.state_dict(), f"../models/{episode}_ppo_model.pth")


# --- Visualization
plt.figure(figsize=(6, 6))
plt.imshow(position_heatmap, origin='lower', cmap='hot')
plt.gca().invert_yaxis()
plt.colorbar(label='Visit Frequency')
plt.title('Agent Exploration Heatmap')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(episode_rewards, label='Total (extrinsic + intrinsic)')
plt.plot(extrinsic_episode_rewards, label='Extrinsic Only')
plt.plot(intrinsic_episode_rewards, label='Intrinsic Only')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Curve')
plt.legend()
plt.show()
