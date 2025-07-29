import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from src.envs.action_wrapper import SaveActionWrapper
from src.envs.observation_wrapper import SaveObsWrapper
from src.envs.reward_wrapper import IntrinsicRewardWrapper


def make_env(id, model, cfg, render_mode=None):
    env = gym.make(id, render_mode=render_mode, max_steps=cfg.max_steps)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    act_env = SaveActionWrapper(env, cfg.allowed_actions)
    obs_env = SaveObsWrapper(act_env)
    reward_env = IntrinsicRewardWrapper(obs_env, act_env, model, cfg)
    return reward_env
