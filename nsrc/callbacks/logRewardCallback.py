from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class LogIntrinsicExtrinsicRewardsCallback(BaseCallback):
    def __init__(self, reward_wrapper, verbose=0):
        super().__init__(verbose)
        self.reward_wrapper = reward_wrapper
        self.extrinsic_rewards_per_episode = []
        self.intrinsic_rewards_per_episode = []

    def _on_step(self) -> bool:
        dones = self.locals["dones"]  # von VecEnv, Liste der Envs
        for env_idx, done in enumerate(dones):
            if done:
                extrinsic_sum = np.sum(self.reward_wrapper.extrinsic_rewards)
                intrinsic_sum = np.sum(self.reward_wrapper.intrinsic_rewards)
                self.extrinsic_rewards_per_episode.append(extrinsic_sum)
                self.intrinsic_rewards_per_episode.append(intrinsic_sum)

                # Tensorboard Logging
                self.logger.record(f"rollout/ext_rew_episode_{env_idx}", extrinsic_sum)
                self.logger.record(f"rollout/int_rew_episode_{env_idx}", intrinsic_sum)
                self.reward_wrapper.reset_reward_buffers()
        return True

    def _on_training_end(self):
        # Optional: zum Debuggen nach dem Training
        print("Extrinsic Episode Rewards:", self.extrinsic_rewards_per_episode)
        print("Intrinsic Episode Rewards:", self.intrinsic_rewards_per_episode)
