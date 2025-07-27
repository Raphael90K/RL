from dataclasses import dataclass
import random
import numpy as np
import torch


@dataclass
class Config:
    seed: int = 42
    log_dir: str = "../logs"
    save_dir: str = "../models"
    tensorboard_log: str = "../ppo_tensorboard"
    env_name: str = "MiniGrid-MultiRoom-N2-S4-v0"
    max_steps: int = 128
    verbose: int = 1
    save_freqency: int = 100_000
    num_envs: int = 1

    # PPO configuration
    model_lr: float = 1e-4
    batch_size: int = 256
    n_steps: int = 2048
    n_epochs: int = 4
    ent_coef: float = 0.0005
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cuda"
    total_timesteps: int = 5_000_000

    # Environment configuration
    frame_stack_size: int = 1
    eta_intrinsic: float = 0.05
    norm_intrinsic: bool = True
    norm_ema_decay: float = 0.99
    norm_ema_eps: float = 1e-4
    action_dim: int = 4
    allowed_actions: tuple = tuple([0, 1, 2, 5])

    # RND configuration
    rnd_lr: float = 1e-4

    # ICM configuration
    icm_lr: float = 1e-4
    icm_beta: float = 0.2

    # BYOL configuration
    byol_ema_decay: float = 0.99

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
