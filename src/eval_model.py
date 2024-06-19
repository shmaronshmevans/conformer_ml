import sys
import os

sys.path.append("..")

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import torch.multiprocessing as mp
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


def predict_model(data_loader, model, rank):

    output = torch.tensor([]).to(rank)
    model.eval()
    # Test a deep learning model on a given dataset and compute the test loss.
    num_batches = len(data_loader)
    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        # output[0] = convolution
        # output[1] = transformer
        X, y = X.to(rank), y.to(rank)
        # Forward pass to obtain model predictions.
        y_star = model(X)
        gc.collect()
        output = torch.cat((output, y_star[1]), 0)

    return output


def evaluate(
    train_dataset,
    df_train_ls,
    df_test_ls,
    test_dataset,
    model,
    batch_size,
    title,
    features,
    rank,
):
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_eval_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    print("loaders successful")

    train_predict = predict_model(train_eval_loader, model, rank)
    test_predict = predict_model(test_eval_loader, model, rank)

    print("predictions input... offloading to cpus")

    train_predict = train_predict.cpu()
    test_predict = test_predict.cpu()

    print("|| Un-normalization ||")
    dist.barrier()
    if rank == 0:
        train_out = output(train_predict, df_train_ls, forecast_hour, past_steps, single)
        test_out = output(test_predict, df_test_ls, forecast_hour, past_steps, single)

        print("saving data")

        dfout = pd.concat([train_out, test_out], axis=0).reset_index().drop(columns="index")
        dfout.to_parquet(
            f"/home/aevans/conformer_ml/src/data/visuals/{clim_div}_ml_output.parquet"
        )

        print("creating plots ")
        plot_outputs(
            dfout, train_predict, stations, today_date, today_date_hr, clim_div, single
        )
        print("finished :)")


def output(prediction, df_ls, forecast_hour, past_steps):
    df_out = pd.DataFrame()
    n = prediction.shape[1]
    i = 0
    print("PREDICT", prediction.shape)
    while n > i:
        target = df_ls[i]["target_error"].tolist()
        target = target[int(len(target) - prediction.shape[0]) :]
        output = prediction[:, i]
        output = output.tolist()
        df_out[f"{i}_transformer_output"] = output
        df_out[f"{i}_target"] = target
        i += 1
    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean
    df_out = df_out.sort_index()

    return df_out


def plot_outputs(df_out, prediction, stations, clim_div):
    import matplotlib.pyplot as plt

    df_out = df_out.sort_index()
    fig, axs = plt.subplots(
        prediction.shape[1], figsize=(21, 21), sharex=True, sharey=True
    )
    n = prediction.shape[1]
    i = 0
    while n > i:
        axs[i].set_ylabel(f"{stations[i]}")
        axs[i].plot(df_out[f"{i}_target"], c="r", label="Target")
        axs[i].plot(
            df_out[f"{i}_conformer_output"],
            c="b",
            alpha=0.7,
            label="Conformer Output",
        )
        i += 1
    fig.suptitle(f"Conformer Output v Target", fontsize=28)
    axs[-1].set_xticklabels([2018, 2019, 2020, 2021, 2022, 2023], fontsize=18)
    axs[-1].set_xticks(
        np.arange(0, len(df_out["0_target"]), (len(df_out["0_target"])) / 6)
    )
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(f"/home/aevans/conformer_ml/src/data/visuals/{clim_div}_output.png")


def eval_main(BATCH_SIZE, CLIM_DIV, forecast_hour, past_timesteps, model_path, rank, world_size, single=False):
    torch.manual_seed(101)
    setup(rank, world_size)

    # Use GPU if available
    device = rank % torch.cuda.device_count()
    print("I'm using device: ", device)
    today_date, today_date_hr = get_time_title.get_time_title(CLIM_DIV)

    # create data
    print("creating data")
    train_df, test_df, train_ims, test_ims, target, stations = (
        read_in_images.create_data_for_model(CLIM_DIV)
    )
    print("data curated")

    # load datasets
    train_dataset = ImageSequenceDataset(train_ims, train_df, target, past_timesteps, forecast_hour)
    test_dataset = ImageSequenceDataset(test_ims, test_df, target, past_timesteps, forecast_hour)

    # define model parameters
    ml = model2.Conformer(
        patch_size=16,
        in_chans=800,
        stations=len(stations),
        past_timesteps=past_timesteps,
        forecast_hour=forecast_hour,
        pos_embedding=0.65,
        num_layers=1,
    )
    if torch.cuda.is_available():
        ml.cuda()

    ml.load_state_dict(torch.load(model_path))

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

    print("-- begin evaluation --")
    evaluate(
        train_dataset,
        df_train_ls,
        df_test_ls,
        test_dataset,
        ml,
        BATCH_SIZE,
        today_date_hr,
        features,
        rank,
    )
    cleanup()

WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
RANK = int(os.environ["RANK"]) if "RANK" in os.environ else -1

eval_main(
    BATCH_SIZE=1,
    CLIM_DIV="Mohawk Valley",
    forecast_hour=4,
    past_timesteps=32,
    model_path="/home/aevans/conformer_ml/src/data/temp_df/Mohawk Valley_20240517_23:06.pth",
    rank=RANK,
    world_size = WORLD_SIZE
)
