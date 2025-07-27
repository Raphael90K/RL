import gymnasium as gym
from datetime import datetime

from gymnasium import ActionWrapper
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.callbacks.logPlainRewardCallback import LogExtrinsicRewardPlainCallback

from src.callbacks.uniquePositionCallback import UniquePositionCallback
from src.envs.action_wrapper import SaveActionWrapper


def train_plain(cfg):
    name = 'PLAIN'
    env = gym.make(cfg.env_name, max_steps=cfg.max_steps)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = SaveActionWrapper(env, cfg.allowed_actions)
    env = DummyVecEnv([lambda: Monitor(env, f'{cfg.log_dir}/{name}')])
    env.seed(cfg.seed)
    env.action_space = gym.spaces.discrete.Discrete(cfg.action_dim)

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=cfg.verbose,
        tensorboard_log=f"{cfg.tensorboard_log}/{name}",
        device=cfg.device,
        ent_coef=cfg.ent_coef,
        gamma=cfg.gamma,
        learning_rate=cfg.model_lr,
        n_epochs=cfg.n_epochs,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        gae_lambda=cfg.gae_lambda,
    )
    ### Callbacks

    unique_pos_callback = UniquePositionCallback()
    rewards_callback = LogExtrinsicRewardPlainCallback()
    save_callback = CheckpointCallback(cfg.save_freqency, save_path=f'{cfg.save_dir}/{name}_{cfg.env_name}',
                                       name_prefix=f"{name}_checkpoint")

    callbacks = CallbackList([unique_pos_callback, rewards_callback, save_callback])
    model.learn(cfg.total_timesteps, callback=callbacks, tb_log_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
