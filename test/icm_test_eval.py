import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# --- CNN + LSTM Actor-Critic
class CNNLSTMActorCritic(nn.Module):
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
        x = torch.permute(x, (0, 3, 1, 2))  # Change from (batch, height, width, channels) to (batch, channels, height, width)
        x = x / 255.0
        conv = self.conv(x)
        flat = torch.flatten(conv, 1)
        features = self.fc(flat)
        logits = self.policy(features)
        value_ext = self.value_ext(features)
        value_int = self.value_int(features)
        return logits, value_ext, value_int


# --- Environment
env = gym.make("MiniGrid-FourRooms-v0", render_mode="human", max_steps=150)
env = RGBImgPartialObsWrapper(env)
action_dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained model
model = CNNLSTMActorCritic(action_dim).to(device)
model.load_state_dict(torch.load("../models/300_ppo.pth", map_location=device))
model.eval()

# --- Greedy Evaluation
obs, _ = env.reset()
#hidden = model.init_hidden(batch_size=1, device=device)

obs_img = torch.tensor(obs['image'], dtype=torch.float32, device=device) / 255.0
obs_img = obs_img.unsqueeze(0)

done = False
positions = []
total_reward = 0

while not done:
    logits, vx, vi = model(obs_img)
    action = torch.argmax(logits, dim=-1).item()

    obs, reward, terminated, truncated, _ = env.step(action)

    obs_img = torch.tensor(obs['image'], dtype=torch.float32, device=device) / 255.0
    obs_img = obs_img.unsqueeze(0)

    pos = env.unwrapped.agent_pos
    print(f'position: {pos}, action: {action}, reward: {reward}, logits: {logits}')
    positions.append((int(pos[0]), int(pos[1])))
    total_reward += reward
    done = terminated or truncated

print(f"\nTotal extrinsic reward in test episode: {total_reward}\n")

# --- Plot visited positions
width, height = env.unwrapped.width, env.unwrapped.height
heatmap = np.zeros((height, width))
for x, y in positions:
    heatmap[y, x] += 1

plt.figure(figsize=(6, 6))
plt.imshow((heatmap), origin='lower', cmap='hot')
plt.gca().invert_yaxis()
plt.colorbar(label='Visit Frequency')
plt.title('Agent Path Heatmap (Greedy Policy)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
