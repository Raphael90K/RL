# train.py
import torch
import torch.nn.functional as F
from env.four_rooms import make_four_rooms_env
from agent.agent import QNetwork
from config import config

# Austauschbar:
# from exploration.icm import ICMWrapper as Curiosity
from exploration.rnd import RNDWrapper as Curiosity


device = torch.device("cuda")

def train_step(q_net, target_q_net, optimizer, obs, action, reward, next_obs, done, gamma):
    obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

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

def train(q_net):
    target_q_net = QNetwork(obs_shape, n_actions).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=config["learning_rate"])
    curiosity = Curiosity(obs_shape, n_actions)

    for episode in range(config["episodes"]):
        extrinsic_total = 0
        intrinsic_total = 0

        obs, _ = env.reset()
        done = False

        while not done:
            obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            with torch.no_grad():
                q_values = q_net(obs_tensor).squeeze(0)

            action = torch.argmax(q_values).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            use_intrinsic = episode < config.get("curiosity_episodes", 100)

            if use_intrinsic:
                intrinsic_reward = curiosity.compute_intrinsic_reward(obs, next_obs, action)
                curiosity.update(obs, next_obs, action)
            else:
                intrinsic_reward = 0.0

            combined_reward = reward + intrinsic_reward

            train_step(q_net, target_q_net, optimizer, obs, action, combined_reward, next_obs, done, config["gamma"])

            extrinsic_total += reward
            intrinsic_total += intrinsic_reward
            obs = next_obs

        if episode % config["target_update_freq"] == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode} - Total: {extrinsic_total + intrinsic_total:.2f} | Ext: {extrinsic_total:.2f} | Int: {intrinsic_total:.2f}")

def evaluate_agent(q_net):
    eval_env = make_four_rooms_env(render_mode="human")
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            q_values = q_net(obs_tensor)
            action = q_values.argmax().item()

        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Evaluation Reward: {total_reward:.2f}")
    eval_env.close()


if __name__ == "__main__":
    env = make_four_rooms_env(max_episode_steps=config["max_steps"])
    obs_shape = (3, *env.observation_space.shape[:2])
    n_actions = 3

    q_net = QNetwork(obs_shape, n_actions).to(device)
    q_net.summary()

    train(q_net)
    evaluate_agent(q_net)