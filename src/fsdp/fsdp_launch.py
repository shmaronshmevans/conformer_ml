# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import os
import fsdp_engine
import torch
import argparse
import torch.multiprocessing as mp

# mp.set_start_method('spawn')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(1),
        help="input epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(5),
        help="input batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=float(7e-4),
        help="input learning rate for training",
    )
    parser.add_argument(
        "--clim_div",
        type=str,
        default=str("Mohawk Valley"),
        help="input climate division for training",
    )
    parser.add_argument(
        "--forecast_hour",
        type=int,
        default=int(4),
        help="input forecast hour for training",
    )
    parser.add_argument(
        "--past_timesteps",
        type=int,
        default=int(32),
        help="input past time steps for training",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )

    args, unknown = parser.parse_known_args()

    WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    RANK = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    print("rank", RANK)
    print("World_Size", WORLD_SIZE)

    fsdp_engine.main(RANK, WORLD_SIZE, args)
