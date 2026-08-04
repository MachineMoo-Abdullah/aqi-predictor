import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Feature_store_cloud.connection import engine


def read_features():
    """
    Read all features from the feature store.
    """

    query = """
    SELECT *
    FROM aqi_features
    ORDER BY datetime;
    """

    df = pd.read_sql(query, engine)

    # Convert datetime column
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df


if __name__ == "__main__":

    df = read_features()

    print(df.head())
    print(df.shape)