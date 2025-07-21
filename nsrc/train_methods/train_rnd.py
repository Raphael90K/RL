import gymnasium as gym
from datetime import datetime

import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

from nsrc.config import Config
from nsrc.intrinsic.rnd_model import RNDConvModel, RNDUpdateCallback
from nsrc.callbacks.logRewardCallback import LogIntrinsicExtrinsicRewardsCallback

from nsrc.envs.env import make_env


def train_rnd(cfg: Config):
    obs_buffer = []
    device = torch.device(cfg.device)

    rnd_model = RNDConvModel(obs_buffer).to(device)
    name = 'RND'

    reward_env = make_env(cfg.env_name, rnd_model, cfg)  # Create the environment
    reward_env.action_space = gym.spaces.discrete.Discrete(cfg.action_dim)  # Set action space to Discrete(3) for the environment

    env = DummyVecEnv([lambda: Monitor(reward_env, f'{cfg.log_dir}/{name}')])  # Monitor to track rewards and other metrics

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=cfg.verbose,
        tensorboard_log=f"{cfg.tensorboard_log}/{name}",
        device=cfg.device,
        ent_coef=0.05,
        gamma=cfg.gamma,
        learning_rate=cfg.model_lr,
        n_epochs=cfg.n_epochs,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        seed=cfg.seed

    )
    ### Callbacks
    update_callback = RNDUpdateCallback(rnd_model)
    log_reward_callback = LogIntrinsicExtrinsicRewardsCallback(reward_env)
    save_callback = CheckpointCallback(cfg.save_freqency, save_path=f'{cfg.save_dir}/{name}', name_prefix=f"{name}_checkpoint")

    callbacks = CallbackList([update_callback, log_reward_callback, save_callback])
    model.learn(cfg.total_timesteps, callback=callbacks, tb_log_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
