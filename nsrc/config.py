from dataclasses import dataclass

@dataclass
class Config:

    seed: int = 42
    log_dir: str = "../logs"
    save_dir: str = "../models"
    tensorboard_log: str = "../ppo_tensorboard"
    env_name: str = "MiniGrid-FourRooms-v0"
    max_steps: int = 128
    verbose: int = 1
    save_freqency: int = 100_000

    # PPO configuration
    model_lr: float = 3e-4
    batch_size: int = 256
    n_steps: int = 1024
    n_epochs: int = 5
    ent_coef: float = 0.05
    clip_range: float = 0.2
    gamma: float = 0.99
    device: str = "cuda"
    total_timesteps: int = 5_000_000

    # Environment configuration
    frame_stack_size: int = 4
    beta_intrinsic: float = 1
    norm_intrinsic: bool = True

    # RND configuration
    rnd_lr: float = 1e-5

