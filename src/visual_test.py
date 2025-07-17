import torch
from env.env import make_four_rooms_env
from agent.agent import QNetwork
import time


def visual_test(model_path, obs_shape, n_actions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_four_rooms_env(render_mode="human", max_steps=200)

    q_net = QNetwork(obs_shape, n_actions).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            q_values = q_net(obs_tensor)
            action = q_values.argmax().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        time.sleep(0.2)  # für besser erkennbare Visualisierung

    print(f"Total reward in visual test: {total_reward}")
    env.close()


if __name__ == "__main__":
    # Beispiel für NoCuriosity. Ändere den Pfad, um andere Methoden zu testen.
    model_path = "dqn_RND.pth"
    obs_shape = (20, 7, 7)  # (C, H, W) -> anpassen je nach Umgebung
    n_actions = 3

    visual_test(model_path, obs_shape, n_actions)
