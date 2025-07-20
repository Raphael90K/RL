import gymnasium as gym
from datetime import datetime


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

from nsrc.intrinsic.icm_model import ICMModel, ICMUpdateCallback
from nsrc.envs.reward_wrapper import IntrinsicRewardWrapper
from nsrc.envs.env import make_env


def train_icm():
    obs_buffer = []
    next_obs_buffer = []
    action_buffer = []

    obs_shape = (3 * 4, 56, 56)
    icm_model = ICMModel(obs_shape, obs_buffer, next_obs_buffer, action_buffer , action_dim=3)

    env = make_env()
    reward_env = IntrinsicRewardWrapper(env, icm_model, beta=1.0, obs_buffer=obs_buffer)  # Wrap the environment with intrinsic rewards
    env = DummyVecEnv([lambda : Monitor(reward_env)])

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_rnd_tensorboard/ICM",
        device="cuda"
    )
    callback = ICMUpdateCallback(icm_model, obs_buffer, next_obs_buffer, action_buffer, "./ppo_rnd_tensorboard/ICM", reward_env)
    model.learn(5_000, callback=callback, tb_log_name=f"PPO_ICM_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    model.save("ppo_recurrent_icm")
