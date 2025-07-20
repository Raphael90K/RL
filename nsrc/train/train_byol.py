import gymnasium as gym
from datetime import datetime

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
from sb3_contrib import RecurrentPPO
from nsrc.intrinsic.byol_model import BYOLModel, BYOLUpdateCallback
from nsrc.envs.reward_wrapper import IntrinsicRewardWrapper

from nsrc.envs.env import make_env


def train_byol():
    obs_buffer = []
    next_obs_buffer = []

    obs_shape = (3, 56, 56)
    byol_model = BYOLModel(obs_shape)

    env = make_env()
    env = IntrinsicRewardWrapper(env, byol_model, beta=1.0, obs_buffer=obs_buffer)
    env = Monitor(env)
    env = VecFrameStack(env, n_stack=4)

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_rnd_tensorboard/BYOL",
        device="cuda"
    )
    callback = BYOLUpdateCallback(byol_model, obs_buffer, next_obs_buffer, "./ppo_rnd_tensorboard/BYOL", env)
    model.learn(500_000, callback=callback, tb_log_name=f"PPO_BYOL_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    model.save("ppo_recurrent_byol")
