import sys

sys.path.append("..")

import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import pandas as pd
import numpy as np
from datetime import datetime

# dependencies
from processing import create_data_for_vision
from processing import save_output
from processing import get_time_title
from processing import read_in_images
import gc
import model2


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ImageSequenceDataset(Dataset):
    def __init__(self, image_list, dataframe, target, sequence_length, forecast_hour, transform=None):
        self.image_list = image_list
        self.dataframe = dataframe
        self.transform = transform
        self.sequence_length = sequence_length
        self.target = target
        self.forecast_hour = forecast_hour

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        images = []
        x_start = i
        x_end = i + self.sequence_length
        y_start = x_end
        y_end = y_start + self.forecast_hour

        for j in range(x_start, x_end):
            if j < len(self.image_list):
                img_name = self.image_list[j]
                image = np.load(img_name).astype(np.float32)
                image = image[:, :, 4:]
                if self.transform:
                    image = self.transform(image)
                images.append(torch.tensor(image))
            else:
                pad_image = torch.zeros_like(images[0])
                images.append(pad_image)

        while len(images) < self.sequence_length:
            pad_image = torch.zeros_like(images[0])
            images.insert(0, pad_image)

        images = torch.stack(images)
        images = images.to(torch.float32)

        # Extract target values
        y = self.dataframe[self.target].values[y_start : y_end]
        if len(y) < self.sequence_length:
            pad_width = (self.sequence_length - len(y), 0)
            y = np.pad(y, (pad_width, (0, 0)), "constant", constant_values=0)

        y = torch.tensor(y).to(torch.float32)
        return images, y


def train_model(data_loader, model, optimizer, rank, sampler, epoch, loss_func):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    accum = 32
    ddp_loss = torch.zeros(2).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(int(os.environ["RANK"]) % torch.cuda.device_count()), y.to(
            int(os.environ["RANK"]) % torch.cuda.device_count()
        )

        # Forward pass and loss computation.
        # output[0] = convolution
        # output[1] = transformer
        output = model(X)

        loss = loss_func(output[0], y[:, -1, :])
        loss = loss / accum
        loss.backward()
        if (batch_idx + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Track the total loss and the number of processed samples.
        total_loss += loss.item()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += loss.item()
        gc.collect()
        torch.cuda.empty_cache()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    # Compute the average loss for the current epoch.
    avg_loss = total_loss / num_batches

    # Print the average loss on the master process (rank 0).
    if rank == 0:
        train_loss = ddp_loss[0] / ddp_loss[1]
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, train_loss))

    return avg_loss


def test_model(data_loader, model, rank, epoch, loss_func):
    # Test a deep learning model on a given dataset and compute the test loss.
    num_batches = len(data_loader)
    total_loss = 0

    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    # Initialize an array to store loss values.
    ddp_loss = torch.zeros(3).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(rank % torch.cuda.device_count()), y.to(
            rank % torch.cuda.device_count()
        )
        # Forward pass to obtain model predictions.
        output = model(X)
        # Compute loss and add it to the total loss.
        total_loss += loss_func(output[0], y[:, -1, :]).item()
        # Update aggregated loss values.
        ddp_loss[0] += total_loss
        # ddp_loss[0] += total_loss
        ddp_loss[2] += len(X)
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate the average test loss.
    avg_loss = total_loss / num_batches
    # Synchronize and aggregate loss values in distributed testing.
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    # Print the test loss on the master process (rank 0).
    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print("Test set: Average loss: {:.4f}\n".format(avg_loss))

    return avg_loss


def main(rank, world_size, args, single=False):
    if rank == 0:
        experiment = Experiment(
            api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
            project_name="conformer_beta",
            workspace="shmaronshmevans",
        )
    torch.manual_seed(101)
    setup(rank, world_size)

    # Use GPU if available
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print("I'm using device: ", device)
    today_date, today_date_hr = get_time_title.get_time_title(args.clim_div)

    # create data
    train_df, test_df, train_ims, test_ims, target, stations = (
        read_in_images.create_data_for_model(args.clim_div)
    )

    # load datasets
    train_dataset = ImageSequenceDataset(train_ims, train_df, target, args.past_timesteps, args.forecast_hour)
    test_dataset = ImageSequenceDataset(test_ims, test_df, target, args.past_timesteps, args.forecast_hour)
    sequence_length = int(args.past_timesteps + args.forecast_hour)

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {
        "batch_size": args.batch_size,
        "sampler": sampler1,
        "pin_memory": False,
        "num_workers": 4,
    }
    test_kwargs = {
        "batch_size": args.batch_size,
        "sampler": sampler2,
        "pin_memory": False,
        "num_workers": 4,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    hyper_params = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "clim_div": str(args.clim_div),
        "forecast_hr": args.forecast_hour,
        "seq_length": int(args.forecast_hour + args.past_timesteps),
    }

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # define model parameters
    ml = model2.Conformer(
        patch_size=16,
        in_chans=800,
        stations=len(stations),
        past_timesteps=args.past_timesteps,
        forecast_hour=args.forecast_hour,
        pos_embedding=0.65,
        num_layers=1,
    ).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    ml = FSDP(
        ml,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
    )

    # Adam Optimizer
    optimizer = torch.optim.Adam(ml.parameters(), lr=args.learning_rate)
    # MSE Loss
    loss_func = nn.MSELoss()
    # loss_func = FocalLossV3()
    scheduler = StepLR(optimizer, step_size=1)
    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    # early_stopper = EarlyStopper(20)

    for ix_epoch in range(1, args.epochs + 1):
        print("Epoch", ix_epoch)
        train_loss = train_model(
            train_loader,
            ml,
            optimizer,
            rank,
            sampler1,
            ix_epoch,
            loss_func,
        )
        test_loss = test_model(test_loader, ml, rank, ix_epoch, loss_func)
        scheduler.step()
        print()
        if rank == 0:
            train_loss_ls.append(train_loss)
            test_loss_ls.append(test_loss)
            experiment.set_epoch(ix_epoch)
            experiment.log_metric("test_loss", test_loss)
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metrics(hyper_params, epoch=ix_epoch)

    init_end_event.record()
    torch.cuda.synchronize()

    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
        states = ml.state_dict()

        if rank == 0:
            torch.save(
                states,
                f"/home/aevans/conformer_ml/src/data/temp_df/{args.clim_div}_{today_date_hr}.pth",
            )

    print("Successful Experiment")
    if rank == 0:
        # Seamlessly log your Pytorch model
        log_model(experiment, ml, model_name="v5")
        experiment.end()
    print("... completed ...")
    torch.cuda.synchronize()
    cleanup()
    exit()
