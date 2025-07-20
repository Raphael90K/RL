import gymnasium as gym


class SaveActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = None

    def action(self, action):
        self.last_action = action
        return action
