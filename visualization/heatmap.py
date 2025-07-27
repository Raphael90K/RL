import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------------
# Konfiguration
# -----------------------------

BASE_MODEL_DIR = "../models"
AGENT_NAMES = ["ICM", "RND", "BYOL"]
NUM_EPISODES = 100
MAX_STEPS = 50
SEED = 42
GRID_SIZE = 15  # Für konsistente Umgebung

# -----------------------------
# Hilfsfunktionen
# -----------------------------

def make_env(seed):
    env = gym.make("MiniGrid-FourRooms-v0", render_mode=None, max_steps=MAX_STEPS)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env.action_space = gym.spaces.Discrete(3)
    env.reset(seed=seed)
    return env

def collect_visitation(model_path, num_episodes=NUM_EPISODES):
    model = RecurrentPPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(SEED)
    height, width = env.unwrapped.grid.height, env.unwrapped.grid.width
    MAP_SIZE = (height, width)

    visitation = np.zeros(MAP_SIZE, dtype=np.int32)
    vec_env = DummyVecEnv([lambda: env])
    model.set_env(vec_env)

    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)

    for ep in range(num_episodes):
        done = False
        step = 0
        obs = vec_env.reset()
        lstm_states = None
        episode_starts[:] = True

        while not done and step < MAX_STEPS:
            agent_pos = vec_env.envs[0].unwrapped.agent_pos
            x, y = agent_pos
            if 0 <= x < MAP_SIZE[1] and 0 <= y < MAP_SIZE[0]:
                visitation[y, x] += 1

            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_starts = dones
            done = dones[0]
            step += 1

    return visitation / visitation.sum()

def find_checkpoints():
    """
    Findet alle Checkpoints in ../models/{AGENT} und sortiert sie nach Steps.
    Rückgabe: { "ICM": {100000: path, 500000: path}, ... }
    """
    all_paths = {}
    for agent in AGENT_NAMES:
        agent_dir = os.path.join(BASE_MODEL_DIR, agent)
        step_paths = {}
        for fname in os.listdir(agent_dir):
            match = re.search(r"_(\d+)_steps\.zip$", fname)
            if match:
                steps = int(match.group(1))
                full_path = os.path.join(agent_dir, fname)
                step_paths[steps] = full_path
        all_paths[agent] = dict(sorted(step_paths.items()))
    return all_paths

# -----------------------------
# Plot-Funktion für Grid
# -----------------------------

def plot_heatmaps_grid(data_dict):
    methods = sorted(data_dict.keys())
    all_steps = sorted(set(step for steps in data_dict.values() for step in steps.keys()))
    num_rows = len(all_steps)
    num_cols = len(methods)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))

    for col, method in enumerate(methods):
        for row, step in enumerate(all_steps):
            ax = axes[row, col] if num_rows > 1 else axes[col]
            matrix = data_dict[method].get(step, None)

            if matrix is not None:
                sns.heatmap(matrix, ax=ax, cmap="inferno", square=True, cbar=False,
                            xticklabels=False, yticklabels=False, vmin=0, vmax=0.05)
                ax.invert_yaxis()
            else:
                ax.axis("off")

            if col == 0:
                ax.set_ylabel(f"{step // 1000}k", fontsize=10)
            if row == 0:
                ax.set_title(method, fontsize=12)

    plt.suptitle("State Visitation Heatmaps – FourRooms\n(↓ Steps / → Method)", fontsize=16)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    checkpoints = find_checkpoints()
    all_data = {}

    for agent, step_dict in checkpoints.items():
        all_data[agent] = {}
        for step, path in step_dict.items():
            print(f"▶ {agent} – {step} steps")
            visitation = collect_visitation(path, num_episodes=NUM_EPISODES)
            all_data[agent][step] = visitation

    plot_heatmaps_grid(all_data)
