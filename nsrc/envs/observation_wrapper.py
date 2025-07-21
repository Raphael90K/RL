import gymnasium as gym


class SaveObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.next_obs = None

    def observation(self, obs):
        self.next_obs = obs
        return obs
