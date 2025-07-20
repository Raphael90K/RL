import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from intrinsic.rnd_model import RNDModel


env = gym.make("MiniGrid-Empty-Random-6x6-v0", render_mode=None)
env = RGBImgPartialObsWrapper(env)  # Convert to RGB image observation
env = ImgObsWrapper(env)
model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, ent_coef=0.01)
model.learn(15000)

vec_env = model.get_env()
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
print(mean_reward)

model.save("ppo_recurrent")


