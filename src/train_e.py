import time

import torch
import torch.nn.functional as F
import random
from env.env import make_four_rooms_env
from agent.agent import QNetwork
from config import config

device = torch.device("cuda")

def select_action(q_net, obs, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            q_values = q_net(obs_tensor)
        return q_values.argmax().item()

def train_step(q_net, target_q_net, optimizer, obs, action, reward, next_obs, done, gamma):
    obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
    next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(device)

    q_values = q_net(obs_tensor)
    q_value = q_values[0, action]

    with torch.no_grad():
        next_q_values = target_q_net(next_obs_tensor)
        max_next_q = torch.max(next_q_values)
        target = reward + (0.0 if done else gamma * max_next_q.item())

    target = torch.tensor(target, dtype=torch.float32, device=q_value.device)

    loss = F.mse_loss(q_value, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(q_net, obs_shape, n_actions):
    target_q_net = QNetwork(obs_shape, n_actions).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=config["learning_rate"])

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 1000  # Episoden
    for episode in range(config["episodes"]):
        epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay) * (epsilon_start - epsilon_end))

        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(q_net, obs, epsilon, n_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            train_step(q_net, target_q_net, optimizer, obs, action, reward, next_obs, done, config["gamma"])

            total_reward += reward
            obs = next_obs

        if episode % config["target_update_freq"] == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode} - Total reward: {total_reward:.2f} | ε: {epsilon:.2f}")

def evaluate_agent(q_net):
    eval_env = make_four_rooms_env(render_mode="human")
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            q_values = q_net(obs_tensor)
            action = q_values.argmax().item()

        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(1)

    print(f"Evaluation Reward: {total_reward:.2f}")
    eval_env.close()


if __name__ == "__main__":
    env = make_four_rooms_env(max_steps=config["max_steps"])
    h, w, c = env.observation_space.shape
    obs_shape = (c, h, w)
    print("Observation Shape:", env.observation_space.shape)
    n_actions = 3

    q_net = QNetwork(obs_shape, n_actions).to(device)
    q_net.summary()

    train(q_net, obs_shape, n_actions)
    evaluate_agent(q_net)
