import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from processing import hrrr_data
from processing import nysm_data
from processing import get_error
from processing import normalize
import gc


def which_fold(df, fold):
    length = len(df)
    test_len = int(length * 0.2)
    df_train = pd.DataFrame()

    for n in np.arange(0, 5):
        if n != fold:
            df1 = df.iloc[int(0.2 * n * length) : int(0.2 * (n + 1) * length)]
            df_train = pd.concat([df_train, df1])
        else:
            df_test = df.iloc[int(0.2 * n * length) : int(0.2 * (n + 1) * length)]

    return df_train, df_test


def columns_drop(df):
    df = df.drop(
        columns=[
            "level_0",
            "index_x",
            "index_y",
            "lead time",
            "lsm",
            "station_y",
            "lat",
            "lon",
        ]
    )
    df = df.rename(columns={"station_x": "station"})
    return df


def create_data_for_model(clim_div, today_date, forecast_hour, single):
    """
    This function creates and processes data for a vision transformer machine learning model.

    Returns:
        df_train (pandas DataFrame): A DataFrame for training the machine learning model.
        df_test (pandas DataFrame): A DataFrame for testing the machine learning model.
        features (list): A list of feature names.
    """
    # load nysm data
    nysm_df = nysm_data.load_nysm_data()
    nysm_df.reset_index(inplace=True)
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # load hrrr data
    hrrr_df = hrrr_data.read_hrrr_data(forecast_hour)

    if single == True:
        stations = [clim_div]
    else:
        # Filter data by NY climate division
        nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"
        nysm_cats_df = pd.read_csv(nysm_cats_path)
        nysm_cats_df = nysm_cats_df[nysm_cats_df["climate_division_name"] == clim_div]
        stations = nysm_cats_df["stid"].tolist()

    nysm_df = nysm_df[nysm_df["station"].isin(stations)]
    hrrr_df = hrrr_df[hrrr_df["station"].isin(stations)]

    # need to create a master list for valid_times so that all the dataframes are the same shape
    master_time = hrrr_df["valid_time"].tolist()
    for station in stations:
        hrrr_dft = hrrr_df[hrrr_df["station"] == station]
        nysm_dft = nysm_df[nysm_df["station"] == station]
        times = hrrr_dft["valid_time"].tolist()
        times2 = nysm_dft["valid_time"].tolist()
        result = list(set(times) & set(master_time) & set(times2))
        master_time = result
    master_time_final = master_time

    # Filter NYSM data to match valid times from master-list
    nysm_df_filtered = nysm_df[nysm_df["valid_time"].isin(master_time_final)]
    hrrr_df_filtered = hrrr_df[hrrr_df["valid_time"].isin(master_time_final)]

    df_train_ls = []
    df_test_ls = []
    # merge dataframes so that each row is hrrr + nysm data for the same time step
    # do this for each station individually
    for station in stations:
        print(f"Compiling Data for {station}")
        nysm_df1 = nysm_df_filtered[nysm_df_filtered["station"] == station]
        hrrr_df1 = hrrr_df_filtered[hrrr_df_filtered["station"] == station]

        master_df = hrrr_df1.merge(nysm_df1, on="valid_time")
        master_df = columns_drop(master_df)

        # Calculate the error using NWP data.
        master_df = get_error.nwp_error("t2m", master_df)

        # encode for day_of_year
        master_df = normalize.encode(master_df, "day_of_year", 366)

        cols_to_carry = ["valid_time", "station"]
        master_df.to_parquet(
            f"/home/aevans/transformer_ml/src/data/temp_df/{today_date}/{clim_div}/{clim_div}_{station}.parquet"
        )

        new_df = master_df.drop(columns=cols_to_carry)

        # for c in new_df.columns:
        #     new_df[f'{c}_dt'] = new_df[c].diff()
        # new_df = new_df.fillna(0)
        new_df, features = normalize.normalize_df(new_df)
        new_df = new_df.fillna(0)

        # Split the data into training and testing sets.
        df_train, df_test = which_fold(new_df, 3)
        print("Test Set Fraction", len(df_test) / len(new_df))

        df_train_ls.append(df_train)
        df_test_ls.append(df_test)

        print("train_shape", df_train.shape)
        print("test_shape", df_test.shape)
        gc.collect()
    return df_train_ls, df_test_ls, features, stations
