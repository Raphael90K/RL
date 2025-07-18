import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



# --- PPO Actor-Critic Network
class CNNActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*128*16, 128), nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = self.conv(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value



# --- RND Networks
class RNDCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*128*16, 128), nn.ReLU()
        )
        self.target = nn.Sequential(nn.Linear(128, out_dim))
        self.predictor = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        features = self.conv(x)
        target_out = self.target(features)
        pred_out = self.predictor(features)
        return target_out, pred_out



# --- Preprocess observation (use x, y position)
def preprocess(env):
    agent_pos = env.unwrapped.agent_pos
    return torch.tensor(agent_pos, dtype=torch.float32) / 10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Hyperparameters
seed=42
goal_reward = 100
gamma = 0.99
lr = 2.5e-4
intrinsic_reward_scale = 1
clip_epsilon = 0.2
ppo_epochs = 4
temperature = 1
epochs = 50
max_steps = 500

# --- Environment
env = gym.make("MiniGrid-Empty-16x16-v0", render_mode=None, max_steps=500)
env = RGBImgObsWrapper(env)
action_dim = 3
ppo_model = CNNActorCritic(action_dim).to(device)
rnd_model = RNDCNN().to(device)
optimizer_ppo = optim.Adam(ppo_model.parameters(), lr=lr)
optimizer_rnd = optim.Adam(rnd_model.predictor.parameters(), lr=lr)
optimizer_rnd = optim.Adam(rnd_model.predictor.parameters(), lr=lr)
width, height = env.unwrapped.width, env.unwrapped.height


# --- For visualization: track positions
position_heatmap = np.zeros((height, width))

# --- Running Mean Std für Intrinsic Reward vorbereiten
episode_rewards = []
global_min = float('inf')
global_max = float('-inf')


# --- Training Loop
for episode in range(epochs):
    obs, _ = env.reset(seed=seed)
    obs_img = torch.tensor(obs['image'], dtype=torch.float32, device=device) / 255.0  # Normiere RGB [0-1]
    obs_img = obs_img.unsqueeze(0)
    agent_pos = env.unwrapped.agent_pos
    done = False
    log_probs, values, rewards, intrinsic_rewards, actions, states = [], [], [], [], [], []
    positions = []

    while not done:
        positions.append((int(agent_pos[0]), int(agent_pos[1])))
        logits, value = ppo_model(obs_img)
        dist = torch.distributions.Categorical(logits=logits / temperature)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        if reward > 0:
            reward = goal_reward  # Normalize positive rewards to 1
        next_obs_img = torch.tensor(next_obs['image'], dtype=torch.float32, device=device) / 255.0
        next_obs_img = next_obs_img.unsqueeze(0)

        done = terminated or truncated
        agent_pos = env.unwrapped.agent_pos

        target_feature, pred_feature = rnd_model(next_obs_img)
        intrinsic_reward = (target_feature - pred_feature).pow(2).mean().item()

        states.append(obs_img.squeeze(0))
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        intrinsic_rewards.append(intrinsic_reward)

        obs_img = next_obs_img

    # --- Normalize Intrinsic Rewards
    intrinsic_rewards_np = np.array(intrinsic_rewards)
    global_min = min(global_min, intrinsic_rewards_np.min())
    global_max = max(global_max, intrinsic_rewards_np.max())

    if global_max - global_min < 1e-8:
        normalized_intrinsic_rewards = np.zeros_like(intrinsic_rewards_np)
    else:
        normalized_intrinsic_rewards = (intrinsic_rewards_np - global_min) / (global_max - global_min)


    for x, y in positions:
        position_heatmap[y, x] += 1

    returns = []
    G = 0
    for ext_r, int_r in zip(reversed(rewards), reversed(normalized_intrinsic_rewards)):
        G = ext_r + intrinsic_reward_scale * int_r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    log_probs = torch.stack(log_probs).to(device)
    values = torch.stack(values).squeeze().to(device)
    advantage = returns - values.detach()
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    extrinsic_sum = sum(rewards)
    intrinsic_sum = intrinsic_reward_scale * sum(normalized_intrinsic_rewards) / max_steps
    episode_total_reward = extrinsic_sum + intrinsic_sum
    episode_rewards.append(episode_total_reward)

    for _ in range(ppo_epochs):
        logits, values_new = ppo_model(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs_new = dist.log_prob(actions)
        ratios = (log_probs_new - log_probs.detach()).exp()
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - values_new.squeeze()).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss
        optimizer_ppo.zero_grad()
        loss.backward()
        optimizer_ppo.step()

    rnd_loss = 0
    for state in states:
        target, pred = rnd_model(state.unsqueeze(0).to(device))
        rnd_loss += (target - pred).pow(2).mean()
    rnd_loss /= len(states)
    optimizer_rnd.zero_grad()
    rnd_loss.backward()
    optimizer_rnd.step()

    print(f"Episode {episode + 1} complete.")
    print(f"  Extrinsic Reward: {extrinsic_sum:.8f}")
    print(f"  Intrinsic Reward (scaled): {intrinsic_sum:.8f}")
    print(f"  Total Reward: {episode_total_reward:.8f}")

# --- Plot Exploration Heatmap
plt.imshow(obs_img.squeeze(0).to("cpu").detach().numpy())
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(position_heatmap, origin='lower', cmap='hot')
plt.colorbar(label='Visit Frequency')
plt.title('Agent Exploration Heatmap (FourRooms)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# --- Plot Learning Curve
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward (Extrinsic + Intrinsic)')
plt.title('Learning Curve')
plt.show()

torch.save(ppo_model.state_dict(), "ppo_model.pth")