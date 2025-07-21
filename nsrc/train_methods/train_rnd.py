import gymnasium as gym
from datetime import datetime

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from sb3_contrib import RecurrentPPO

from nsrc.config import Config
from nsrc.envs.action_wrapper import SaveActionWrapper
from nsrc.envs.observation_wrapper import SaveObsWrapper
from nsrc.intrinsic.rnd_model import RNDConvModel, RNDUpdateCallback
from nsrc.envs.reward_wrapper import IntrinsicRewardWrapper
from nsrc.callbacks.logRewardCallback import LogIntrinsicExtrinsicRewardsCallback

from nsrc.envs.env import make_env


def train_rnd(cfg: Config):
    obs_buffer = []
    rnd_model = RNDConvModel()
    name = 'RND'

    env = make_env(cfg.env_name, cfg.max_steps)  # Create the environment
    obs_env = SaveObsWrapper(env)
    act_env = SaveActionWrapper(env)
    reward_env = IntrinsicRewardWrapper(obs_env, act_env, rnd_model, obs_buffer,
                                        beta=cfg.beta_intrinsic,
                                        frame_stack_size=cfg.frame_stack_size,
                                        norm=cfg.norm_intrinsic)
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
    update_callback = RNDUpdateCallback(rnd_model, obs_buffer, "./ppo_rnd_tensorboard/RND", reward_env)
    log_reward_callback = LogIntrinsicExtrinsicRewardsCallback(reward_env)
    save_callback = CheckpointCallback(cfg.save_freqency, save_path=f'{cfg.save_dir}/{name}', name_prefix=f"{name}_checkpoint")

    callbacks = CallbackList([update_callback, log_reward_callback, save_callback])

    model.learn(cfg.total_timesteps, callback=callbacks, tb_log_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
