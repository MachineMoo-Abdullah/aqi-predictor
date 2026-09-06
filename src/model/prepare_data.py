import pandas as pd


def add_daily_avg_aqi(df, aqi_column="AQI"):


    df = df.copy()

    df[aqi_column] = pd.to_numeric(
        df[aqi_column],
        errors="coerce"
    )

    df["daily_avg_AQI"] = (
        df[aqi_column]
        .rolling(window=24, min_periods=24)
        .mean()
        .shift(-23)
    )
    df = df.dropna(
        subset=["daily_avg_AQI"]
    ).reset_index(drop=True)
    print(df.head)
    return df

