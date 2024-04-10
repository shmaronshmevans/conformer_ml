import statistics as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


def output(prediction, df_ls, forecast_hour, past_steps):
    df_out = pd.DataFrame()
    n = prediction.shape[1]
    i = 0
    print("PREDICT", prediction.shape)
    while n > i:
        target = df_ls[i]["target_error"].tolist()
        target = target[int(len(target) - prediction.shape[0]) :]
        output = prediction[:, i, -1]
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


def plot_outputs(df_out, prediction, stations, today_date, today_date_hr, clim_div):
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
            df_out[f"{i}_transformer_output"],
            c="b",
            alpha=0.7,
            label="Transformer Output",
        )
        i += 1
    fig.suptitle(f"Transformer Output v Target", fontsize=28)
    axs[-1].set_xticklabels([2018, 2019, 2020, 2021, 2022, 2023], fontsize=18)
    axs[-1].set_xticks(
        np.arange(0, len(df_out["0_target"]), (len(df_out["0_target"])) / 6)
    )
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(
        f"/home/aevans/transformer_ml/src/data/visuals/{today_date}/{clim_div}/{today_date_hr}_output.png"
    )


def predict(data_loader, model, device):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output


def eval_model(
    train_loader,
    test_loader,
    model,
    device,
    df_train_ls,
    df_test_ls,
    stations,
    today_date,
    today_date_hr,
    clim_div,
    forecast_hour,
    past_steps,
):
    train_predict = predict(train_loader, model, device)
    test_predict = predict(test_loader, model, device)

    train_predict = train_predict.cpu()
    test_predict = test_predict.cpu()

    train_out = output(train_predict, df_train_ls, forecast_hour, past_steps)
    test_out = output(test_predict, df_test_ls, forecast_hour, past_steps)

    dfout = pd.concat([train_out, test_out], axis=0).reset_index().drop(columns="index")
    dfout.to_parquet(
        f"/home/aevans/transformer_ml/src/data/temp_df/{today_date}/{clim_div}/{today_date_hr}_ml_output.parquet"
    )
    plot_outputs(dfout, train_predict, stations, today_date, today_date_hr, clim_div)
