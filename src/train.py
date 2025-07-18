import numpy as np
from config import config
import torch


def train(agent, curiosity, env, method_name):
    reward_log = []

    for episode in range(config["episodes"]):
        extrinsic_total = 0
        intrinsic_total = 0
        obs, _ = env.reset()
        done = False
        epsilon = max(0.01, 0.9 - episode / config["curiosity_episodes"])

        while not done:
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            intrinsic_reward = curiosity.compute_intrinsic_reward(obs, next_obs, action)
            curiosity.update(obs, next_obs, action)
            intrinsic_reward = np.clip(intrinsic_reward, 0, 1)

            combined_reward = reward + intrinsic_reward
            agent.train_step(obs, action, combined_reward, next_obs, done, config["gamma"])

            extrinsic_total += reward
            intrinsic_total += intrinsic_reward
            obs = next_obs

        reward_log.append(extrinsic_total)

        if episode % config["target_update_freq"] == 0:
            agent.update_target()

        print(f"[{method_name}] Episode {episode} - Total: {extrinsic_total:.2f} | Int: {intrinsic_total:.2f} | ε: {epsilon:.2f}")

    agent.save(f"dqn_{method_name}.pth")
    return reward_log
