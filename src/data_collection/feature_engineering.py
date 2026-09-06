import pandas as pd
import numpy as np


def _add_time_and_cyclic_features(df):
    """Shared logic — used by both live and historical paths."""
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df = df.drop(columns=['hour', 'dayofweek', 'maonth'])
    return df


def engineer_historical_features(df):
    """Batch path — used for backfill. AQI_change_rate is a simple
    pct_change since the full history is already present."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df = _add_time_and_cyclic_features(df)

    return df


def engineer_live_features(weather, air, last_known_aqi=None):
    """Live path — called every hour with a single API response.
    last_known_aqi must be fetched from the feature store (the most
    recent stored row) since a single row has no history to diff against.
    """
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
        "AQI": air["AQI"],
    }])

    df = _add_time_and_cyclic_features(df)


    return df