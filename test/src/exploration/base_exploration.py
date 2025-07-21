class CuriosityModule:
    def __init__(self, observation_space, action_space):
        pass

    def compute_intrinsic_reward(self, obs, next_obs, action):
        return 0.0  # Standard: Kein Intrinsic Reward

    def update(self, obs, next_obs, action):
        pass  # Wird für ICM, RND etc. genutzt
