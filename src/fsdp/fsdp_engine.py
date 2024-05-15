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

# dependencies
from processing import create_data_for_vision
from processing import save_output
from processing import get_time_title
import gc
import model2


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class MultiStationDataset(Dataset):
    def __init__(
        self, dataframes, target, features, past_steps, future_steps, nysm_vars=14
    ):
        """
        dataframes: list of station dataframes like in the SequenceDataset
        target: target error
        features: list of features for model
        sequence_length: int
        """
        self.dataframes = dataframes
        self.features = features
        self.target = target
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.nysm_vars = nysm_vars

    def __len__(self):
        shaper = min(
            [
                self.dataframes[i].values.shape[0]
                - (self.past_steps + self.future_steps)
                for i in range(len(self.dataframes))
            ]
        )
        return shaper

    def __getitem__(self, i):
        # this is the preceeding sequence_length timesteps
        x = torch.stack(
            [
                torch.tensor(
                    dataframe[self.features].values[
                        i : (i + self.past_steps + self.future_steps)
                    ]
                )
                for dataframe in self.dataframes
            ]
        ).to(torch.float32)
        # stacking the sequences from each dataframe along a new axis, so the output is of shape (batch, stations (len(self.dataframes)), past_steps, features)
        y = torch.stack(
            [
                torch.tensor(
                    dataframe[self.target].values[
                        i + self.past_steps : i + self.past_steps + self.future_steps
                    ]
                )
                for dataframe in self.dataframes
            ]
        ).to(torch.float32)

        # this is (stations, seq_len, features)
        x[:, -self.future_steps :, -self.nysm_vars :] = (
            -999.0
        )  # check that this is setting the right positions to this value

        return x, y


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

        loss = loss_func(output[1], y[:, :, -1])
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
        total_loss += loss_func(output[1], y[:, :, -1]).item()
        # Update aggregated loss values.
        ddp_loss[0] += total_loss
        # ddp_loss[0] += total_loss
        ddp_loss[2] += len(X)
        gc.collect()

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
    df_train_ls, df_test_ls, features, stations = (
        create_data_for_vision.create_data_for_model(
            args.clim_div, today_date, args.forecast_hour, single
        )
    )

    # load datasets
    train_dataset = MultiStationDataset(
        df_train_ls, "target_error", features, args.past_timesteps, args.forecast_hour
    )
    test_dataset = MultiStationDataset(
        df_test_ls, "target_error", features, args.past_timesteps, args.forecast_hour
    )

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
        in_chans=len(features),
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
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
        states = model.state_dict()

        if rank == 0:
            torch.save(
                states,
                f"/home/aevans/conformer_ml/src/data/temp_df/{args.clim_div}_{today_date_hr}.pth",
            )

    print("Successful Experiment")
    if rank == 0:
        # Seamlessly log your Pytorch model
        log_model(experiment, model, model_name="v5")
        experiment.end()
    print("... completed ...")
    torch.cuda.synchronize()
    cleanup()
