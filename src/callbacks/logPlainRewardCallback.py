from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Callback to log extrinsic rewards per episode
class LogExtrinsicRewardPlainCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.extrinsic_rewards_sum = 0
        self.extrinsic_rewards_per_episode = []
        self.intrinsic_rewards_per_episode = []

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"][0]
        self.extrinsic_rewards_sum += rewards
        dones = self.locals["dones"]  # von VecEnv, Liste der Envs
        for env_idx, done in enumerate(dones):
            if done:
                self.extrinsic_rewards_per_episode.append(self.extrinsic_rewards_sum)

                # Tensorboard Logging
                self.logger.record(f"rollout/ext_rew_episode_{env_idx}", self.extrinsic_rewards_sum)
                self.extrinsic_rewards_sum = 0
        return True

    def _on_training_end(self):
        # Optional: zum Debuggen nach dem Training
        print("Extrinsic Episode Rewards:", self.extrinsic_rewards_per_episode)
        print("Intrinsic Episode Rewards:", self.intrinsic_rewards_per_episode)
