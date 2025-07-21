import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from nsrc.envs.action_wrapper import SaveActionWrapper
from nsrc.envs.observation_wrapper import SaveObsWrapper
from nsrc.envs.reward_wrapper import IntrinsicRewardWrapper


def make_env(id, model, cfg, render_mode=None):
    env = gym.make(id, render_mode=render_mode, max_steps=cfg.max_steps)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    act_env = SaveActionWrapper(env)
    obs_env = SaveObsWrapper(act_env)
    reward_env = IntrinsicRewardWrapper(obs_env, act_env, model,
                                        beta=cfg.beta_intrinsic,
                                        frame_stack_size=cfg.frame_stack_size,
                                        norm=cfg.norm_intrinsic)
    return reward_env
