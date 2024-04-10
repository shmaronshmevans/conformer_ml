import statistics as st
import numpy as np
import pandas as pd
import datetime as dt


def encode(data, col, max_val):
    data["day_of_year"] = data["valid_time"].dt.dayofyear
    sin = np.sin(2 * np.pi * data[col] / max_val).astype(float)
    data.insert(loc=(23), column=f"{col}_sin", value=sin)
    cos = np.cos(2 * np.pi * data[col] / max_val)
    data.insert(loc=(23), column=f"{col}_cos", value=cos)
    data = data.drop(columns=["time", "day_of_year"])

    return data


def normalize_df(df):
    for k, r in df.items():
        means = st.mean(df[k])
        stdevs = st.pstdev(df[k])
        df[k] = (df[k] - means) / stdevs

    og_features = [c for c in df.columns if c != "target_error"]
    new_features = og_features
    return df, new_features
