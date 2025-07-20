import gymnasium as gym


class SaveObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None

    def observation(self, obs):
        self.last_obs = obs
        return obs
