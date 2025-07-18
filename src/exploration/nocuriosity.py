from exploration.base_exploration import CuriosityModule


class NoCuriosity (CuriosityModule):
    def compute_intrinsic_reward(self, obs, next_obs, action):
        return 0.0
    def update(self, obs, next_obs, action):
        pass

