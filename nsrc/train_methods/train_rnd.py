import gymnasium as gym
from datetime import datetime

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from sb3_contrib import RecurrentPPO
from nsrc.intrinsic.rnd_model import RNDConvModel, RNDUpdateCallback
from nsrc.envs.reward_wrapper import IntrinsicRewardWrapper

from nsrc.envs.env import make_env


def train_rnd():
    obs_buffer = []
    rnd_model = RNDConvModel()

    env = make_env()
    reward_env = IntrinsicRewardWrapper(env, rnd_model, beta=1.0, obs_buffer=obs_buffer)  # Wrap the environment with intrinsic rewards
    env = DummyVecEnv([lambda : Monitor(reward_env)])

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_rnd_tensorboard/RND",
        device="cuda"
    )
    callback = RNDUpdateCallback(rnd_model, obs_buffer, "./ppo_rnd_tensorboard/RND", reward_env)
    model.learn(5_000, callback=callback, tb_log_name=f"PPO_RND_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    model.save("ppo_recurrent_rnd")
