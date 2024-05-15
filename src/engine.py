import model2

# comet
from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# dependencies
from processing import create_data_for_vision
from processing import save_output
from processing import get_time_title
import gc


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

        # # Fetch the previous timestep x_t
        # if i > 0:
        #     x_t = torch.stack(
        #         [
        #             torch.tensor(
        #                 dataframe[self.features].values[
        #                     (i - 1) : (i + self.past_steps + self.future_steps - 1)
        #                 ]
        #             )
        #             for dataframe in self.dataframes
        #         ]
        #     ).to(torch.float32)
        #     x_t[:, -self.future_steps:, -self.nysm_vars:] = -999.0
        # else:
        #     # If i is 0, there's no previous timestep
        #     x_t = torch.zeros_like(x)

        return x, y


def train_model(data_loader, model, optimizer, device, epoch, loss_func):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    accum = 32

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)

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
        gc.collect()

    # Compute the average loss for the current epoch.
    avg_loss = total_loss / num_batches

    # Print the average loss on the master process (rank 0).
    print("epoch", epoch, "train_loss:", avg_loss)

    return avg_loss


def test_model(data_loader, model, device, epoch, loss_func):
    # Test a deep learning model on a given dataset and compute the test loss.
    num_batches = len(data_loader)
    total_loss = 0

    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)
        # Forward pass to obtain model predictions.
        output = model(X)
        # Compute loss and add it to the total loss.
        total_loss += loss_func(output[1], y[:, :, -1]).item()
        gc.collect()

    # Calculate the average test loss.
    avg_loss = total_loss / num_batches
    print("epoch", epoch, "test_loss:", avg_loss)

    return avg_loss


def main(
    EPOCHS, BATCH_SIZE, LEARNING_RATE, CLIM_DIV, forecast_hour, past_timesteps, single
):
    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="conformer_beta",
        workspace="shmaronshmevans",
    )
    torch.manual_seed(101)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("I'm using device: ", device)
    today_date, today_date_hr = get_time_title.get_time_title(CLIM_DIV)
    # create data
    df_train_ls, df_test_ls, features, stations = (
        create_data_for_vision.create_data_for_model(
            CLIM_DIV, today_date, forecast_hour, single
        )
    )

    # load datasets
    train_dataset = MultiStationDataset(
        df_train_ls, "target_error", features, past_timesteps, forecast_hour
    )
    test_dataset = MultiStationDataset(
        df_test_ls, "target_error", features, past_timesteps, forecast_hour
    )

    # define model parameters
    ml = model2.Conformer(
        patch_size=16,
        in_chans=len(features),
        stations=len(stations),
        past_timesteps=past_timesteps,
        forecast_hour=forecast_hour,
        pos_embedding=0.65,
        num_layers=1,
    )
    if torch.cuda.is_available():
        ml.cuda()

    # Adam Optimizer
    optimizer = torch.optim.Adam(ml.parameters(), lr=LEARNING_RATE)
    # MSE Loss
    loss_func = nn.MSELoss()
    # loss_func = FocalLossV3()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    hyper_params = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "clim_div": str(CLIM_DIV),
        "forecast_hr": forecast_hour,
        "seq_length": int(forecast_hour + past_timesteps),
    }
    # early_stopper = EarlyStopper(20)

    for ix_epoch in range(1, EPOCHS + 1):
        print("Epoch", ix_epoch)
        train_loss = train_model(
            train_loader, ml, optimizer, device, ix_epoch, loss_func
        )
        test_loss = test_model(test_loader, ml, device, ix_epoch, loss_func)
        print()
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)
        # if early_stopper.early_stop(test_loss):
        #     print(f"Early stopping at epoch {ix_epoch}")
        #     break

    eval_model(
        train_loader,
        test_loader,
        ml,
        device,
        df_train_ls,
        df_test_ls,
        stations,
        today_date,
        CLIM_DIV,
    )
    experiment.end()


main(
    EPOCHS=5,
    BATCH_SIZE=int(5),
    LEARNING_RATE=7e-4,
    CLIM_DIV="Mohawk Valley",
    forecast_hour=4,
    past_timesteps=32,  # fh+past_timesteps needs to be divisible by the number of stations in the clim_div,
    single=False,
)
