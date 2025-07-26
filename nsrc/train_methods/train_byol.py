import gymnasium as gym
from datetime import datetime

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

from src.callbacks.logRewardCallback import LogIntrinsicExtrinsicRewardsCallback
from src.callbacks.uniquePositionCallback import UniquePositionCallback
from src.intrinsic.byol_model import BYOLExploreModel, BYOLExploreUpdateCallback

from src.envs.env import make_env


def train_byol(cfg):
    obs_buffer = []
    next_obs_buffer = []
    act_buffer = []
    act_dim = 3

    obs_shape = (3 * cfg.frame_stack_size, 56, 56)
    byol_model = BYOLExploreModel(obs_shape, act_dim, obs_buffer, next_obs_buffer, act_buffer).to(cfg.device)
    name = 'BYOL'

    reward_env = make_env(cfg.env_name, byol_model, cfg)  # Create the environment
    reward_env.action_space = gym.spaces.discrete.Discrete(
        cfg.action_dim)  # Set action space to Discrete(3) for the environment

    env = DummyVecEnv(
        [lambda: Monitor(reward_env, f'{cfg.log_dir}/{name}')])  # Monitor to track rewards and other metrics

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
    update_callback = BYOLExploreUpdateCallback(byol_model)
    unique_pos_callback = UniquePositionCallback()
    log_reward_callback = LogIntrinsicExtrinsicRewardsCallback(reward_env)
    save_callback = CheckpointCallback(cfg.save_freqency, save_path=f'{cfg.save_dir}/{name}',
                                       name_prefix=f"{name}_checkpoint")

    callbacks = CallbackList([update_callback, unique_pos_callback, log_reward_callback, save_callback])
    model.learn(cfg.total_timesteps, callback=callbacks, tb_log_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
