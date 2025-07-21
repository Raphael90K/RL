import matplotlib.pyplot as plt
from env.env import make_four_rooms_env
from agent.dqn_agent import DQNAgent
from exploration.icm import ICMWrapper
from exploration.rnd import RNDWrapper
from exploration.byol import BYOLWrapper
from exploration.nocuriosity import NoCuriosity
from config import config
from train import train
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run(method_cls, method_name, obs_shape, n_actions):
    env = make_four_rooms_env(max_steps=config["max_steps"])
    agent = DQNAgent(obs_shape, n_actions, device=device)
    curiosity = method_cls(obs_shape, n_actions)
    rewards = train(agent, curiosity, env, method_name)
    return rewards



if __name__ == "__main__":
    env = make_four_rooms_env(max_steps=config["max_steps"])
    h, w, c = env.observation_space.shape
    obs_shape = (c, h, w)
    n_actions = 3

    methods = [
#        (NoCuriosity, "NoCuriosity"),
#        (ICMWrapper, "ICM"),
        (RNDWrapper, "RND"),
#        (BYOLWrapper, "BYOL")
    ]

    results = {}

    for method_cls, name in methods:
        rewards = run(method_cls, name, obs_shape, n_actions)
        results[name] = rewards

    for name, rewards in results.items():
        plt.plot(rewards, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Extrinsic Reward")
    plt.legend()
    plt.title("Learning Curves")
    plt.show()
