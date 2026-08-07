import pandas as pd
import numpy as np


def engineer_features(weather, air):

    df = pd.DataFrame([{

        "temperature_2m": weather["temperature_2m"],
        "relative_humidity_2m": weather["relative_humidity_2m"],
        "surface_pressure": weather["surface_pressure"],
        "precipitation": weather["precipitation"],
        "wind_speed_10m": weather["wind_speed_10m"],

        "datetime": pd.to_datetime(weather["datetime"]),

        "pm2_5": air["pm2_5"],
        "pm10": air["pm10"],
        "carbon_monoxide": air["carbon_monoxide"],
        "nitrogen_dioxide": air["nitrogen_dioxide"],
        "sulphur_dioxide": air["sulphur_dioxide"],
        "ozone": air["ozone"],

        "AQI": air["AQI"]

    }])

    # Time Features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # Cyclic Encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df

import pandas as pd
import numpy as np


def engineer_historical_features(df):
    """
    Apply the same feature engineering used for live data
    to the entire historical dataset.
    """

    # Ensure datetime is datetime type
    df["datetime"] = pd.to_datetime(df["datetime"])

    # --------------------
    # Time Features
    # --------------------
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # --------------------
    # Cyclic Encoding
    # --------------------
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --------------------
    # AQI Change Rate
    # --------------------
    df["AQI_change_rate"] = (
        df["AQI"]
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return df
