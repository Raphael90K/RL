import gymnasium as gym
from gymnasium import spaces


class SaveActionWrapper(gym.ActionWrapper):
    def __init__(self, env, allowed_actions: list):
        super().__init__(env)
        self.last_action = None

        self.allowed_actions = allowed_actions
        self.action_map = {i: a for i, a in enumerate(self.allowed_actions)}
        self.action_space = spaces.Discrete(len(self.allowed_actions))

    def action(self, action):
        self.last_action = action
        return self.action_map[action]
