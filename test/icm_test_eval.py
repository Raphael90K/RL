import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# --- PPO Actor-Critic Network (wie im Training)
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
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


# --- Environment
env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="human", max_steps=500)
env = RGBImgObsWrapper(env)
action_dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained model
model = CNNActorCritic(action_dim).to(device)
model.load_state_dict(torch.load("ppo_model.pth", map_location=device))
model.eval()


# --- Greedy Evaluation
obs, _ = env.reset(seed=42)
obs_img = torch.tensor(obs['image'], dtype=torch.float32, device=device) / 255.0
obs_img = obs_img.unsqueeze(0)

done = False
positions = []
total_reward = 0

while not done:
    logits, _ = model(obs_img)
    action = torch.argmax(logits, dim=-1).item()

    obs, reward, terminated, truncated, _ = env.step(action)
    obs_img = torch.tensor(obs['image'], dtype=torch.float32, device=device) / 255.0
    obs_img = obs_img.unsqueeze(0)

    pos = env.unwrapped.agent_pos
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
plt.imshow(heatmap, origin='lower', cmap='hot')
plt.colorbar(label='Visit Frequency')
plt.title('Agent Path Heatmap (Greedy Policy)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
