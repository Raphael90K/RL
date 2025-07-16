import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper

env = gym.make("MiniGrid-FourRooms-v0")

obs, _ = env.reset()
plt.imshow(obs["image"])
plt.show()

env_obs = RGBImgObsWrapper(env)
obs, _ = env_obs.reset()
plt.imshow(obs["image"])
plt.show()

env_obs = RGBImgPartialObsWrapper(env)
obs= env_obs.observation(obs)
plt.imshow(obs["image"])
plt.show()
