import pandas as pd
from pathlib import Path
import sys
from sqlalchemy import text

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

    with engine.connect() as conn:
        df = pd.read_sql(
            text(query),
            conn
        )

    # Convert datetime column
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df