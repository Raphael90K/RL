import multiprocessing as mp
from train_rnd import train_rnd
from train_icm import train_icm
from train_byol import train_byol


if __name__ == "__main__":
    processes = [
        mp.Process(target=train_rnd),
        mp.Process(target=train_icm),
        mp.Process(target=train_byol),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
