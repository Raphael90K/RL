import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5', render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)

    print(obs)
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
env.close()