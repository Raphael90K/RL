from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class UniquePositionCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.visited_positions = set()

    def _on_step(self) -> bool:
        self.track_position()
        dones = self.locals["dones"]  # von VecEnv, Liste der Envs
        for env_idx, done in enumerate(dones):
            if done:
                unique_positions = len(self.visited_positions)
                self.logger.record(f"rollout/unique_positions_env_{env_idx}", unique_positions)
                self.visited_positions.clear()
        return True

    def track_position(self):
        x, y = self.training_env.unwrapped.get_attr('agent_pos')[0]
        self.visited_positions.add((x, y))
