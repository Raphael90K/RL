from dataclasses import dataclass

@dataclass
class Config:

    seed: int = 42
    log_dir: str = "../logs"
    save_dir: str = "../models"
    tensorboard_log: str = "../ppo_tensorboard"
    env_name: str = "MiniGrid-FourRooms-v0"
    max_steps: int = 64
    verbose: int = 1
    save_freqency: int = 100_000
    num_envs: int = 1

    # PPO configuration
    model_lr: float = 3e-4
    batch_size: int = 128
    n_steps: int = 512
    n_epochs: int = 5
    ent_coef: float = 0.05
    clip_range: float = 0.2
    gamma: float = 0.99
    device: str = "cuda"
    total_timesteps: int = 5_000_000

    # Environment configuration
    frame_stack_size: int = 1
    beta_intrinsic: float = 0.1
    norm_intrinsic: bool = False
    action_dim: int = 3

    # RND configuration
    rnd_lr: float = 1e-5

    # ICM configuration
    icm_lr: float = 1e-5
    icm_beta: float = 0.2

    # BYOL configuration
    ema_decay:float = 0.99
