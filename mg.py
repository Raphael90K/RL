import gymnasium as gym
import minigrid
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

ALLOWED_ACTIONS = [0, 1, 2]  # optional


# Hyperparameter
alpha = 0.1       # Lernrate
gamma = 0.99      # Discount-Faktor
epsilon = 0.1     # Startwert für epsilon
episodes = 1000
max_steps = 500

# Umgebung erstellen
env = gym.make("MiniGrid-FourRooms-v0", render_mode=None, max_episode_steps=max_steps)
action_size = env.action_space.n  # Korrekt: Anzahl der Aktionen (z. B. 7)

# Q-Table mit richtiger Aktionsanzahl
Q = defaultdict(lambda: np.zeros(action_size))


# Hilfsfunktion: Observation zu String (damit man sie als Key nutzen kann)
def obs_to_state(obs):
    # Nimmt nur Agent-Position, Blickrichtung und Objekt in Blickrichtung
    image = obs["image"]
    return tuple(image.flatten())

# Training
rewards = []

for ep in range(episodes):
    if ep > 900:
        env = gym.make("MiniGrid-FourRooms-v0", render_mode="human", max_episode_steps=max_steps)
    obs, info = env.reset()
    state = obs_to_state(obs)
    done = False
    total_reward = 0

    while not done:
        # ε-greedy Action Selection
        if random.random() < epsilon:
            action = random.choice(ALLOWED_ACTIONS)
        else:
            q_vals = Q[state]
            allowed_q = [q_vals[a] for a in ALLOWED_ACTIONS]
            max_q = max(allowed_q)
            best_actions = [a for a in ALLOWED_ACTIONS if q_vals[a] == max_q]
            action = random.choice(best_actions)

        next_obs, reward, terminated, truncated, info = env.step(action)
        if reward > 0:
            print(f"🎯 Goal erreicht! Episode {ep}, Reward: {reward}")
        next_state = obs_to_state(next_obs)
        done = terminated or truncated

        # Q-Learning Update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

    # Optional: Epsilon Decay
    epsilon = max(0.01, epsilon * 0.995)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}: Reward = {total_reward}, ε = {epsilon:.3f}")

env.close()

print(Q)

# Plotten der Rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning mit ε-greedy in MiniGrid")
plt.grid()
plt.show()
