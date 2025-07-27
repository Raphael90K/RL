import multiprocessing as mp

from src.train_methods.train_plain import train_plain
from train_methods.train_rnd import train_rnd
from train_methods.train_icm import train_icm
from train_methods.train_byol import train_byol
from config import Config

if __name__ == "__main__":
    cfg = Config()
    cfg.set_seed()

    processes = [
        mp.Process(target=train_rnd, args=(cfg,)),
        mp.Process(target=train_icm, args=(cfg,)),
        #mp.Process(target=train_byol, args=(cfg,)),
        mp.Process(target=train_plain, args=(cfg,)),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
