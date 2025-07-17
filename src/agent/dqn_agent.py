import torch
import numpy as np
from src.agent.agent import QNetwork
from src.config import config


class DQNAgent:
    def __init__(self, obs_shape, n_actions, device="cuda"):
        self.device = torch.device(device)
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.q_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_q_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config["learning_rate"])

    def act(self, obs, epsilon):
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor).squeeze(0)
        if np.random.rand() < epsilon:
            return np.random.choice(range(self.n_actions))
        return torch.argmax(q_values).item()

    def train_step(self, obs, action, reward, next_obs, done, gamma):
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        q_values = self.q_net(obs_tensor)
        q_value = q_values[0, action]

        with torch.no_grad():
            next_q_values = self.target_q_net(next_obs_tensor)
            max_next_q = torch.max(next_q_values)
            target = reward + (0.0 if done else gamma * max_next_q.item())

        target = torch.tensor(target, dtype=torch.float32, device=self.device)

        loss = torch.nn.functional.mse_loss(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
