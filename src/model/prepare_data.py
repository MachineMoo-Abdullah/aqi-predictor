import pandas as pd


def add_daily_avg_aqi(df, aqi_column="AQI"):
    """
    Add a forward-looking 24-hour average AQI column.

    Example:
        Row 1  -> mean(AQI rows 1-24)
        Row 2  -> mean(AQI rows 2-25)
        Row 3  -> mean(AQI rows 3-26)
        ...

    The last 23 rows are removed because they do not
    have a complete 24-hour window.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the AQI column.

    aqi_column : str
        Name of the AQI column. Default = "AQI".

    Returns
    -------
    pandas.DataFramea
        Dataset with a new "daily_avg_AQI" column.
    """

    df = df.copy()

    # Make sure AQI is numeric
    df[aqi_column] = pd.to_numeric(
        df[aqi_column],
        errors="coerce"
    )

    # Calculate 24-hour forward average
    df["daily_avg_AQI"] = (
        df[aqi_column]
        .rolling(window=24, min_periods=24)
        .mean()
        .shift(-23)
    )

    # Remove rows that don't have a complete
    # 24-hour window
    df = df.dropna(
        subset=["daily_avg_AQI"]
    ).reset_index(drop=True)
    print(df.head)
    return df

