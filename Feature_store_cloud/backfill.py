from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import text

from Feature_store_cloud.connection import engine
from sqlalchemy.types import DateTime


def upload_features(df: pd.DataFrame):
    """
    Upload hourly features to Supabase.
    Creates AQI_change_rate automatically.
    Avoids duplicate timestamps.
    """

    df["datetime"] = pd.to_datetime(df["datetime"])

    timestamp = df.loc[0, "datetime"]

    with engine.begin() as conn:

        # Check whether table exists
        table_exists = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1
                FROM information_schema.tables
                WHERE table_name='aqi_features'
                );
            """)
        ).scalar()

        # First upload
        if not table_exists:

            df["AQI_change_rate"] = 0.0

            df.to_sql(
                name="aqi_features",
                con=conn,
                if_exists="replace",
                index=False
            )

            print("Feature store created successfully.")
            return

        # Duplicate check
        exists = conn.execute(
            text("""
                SELECT 1
                FROM aqi_features
                WHERE datetime = :dt
                LIMIT 1
            """),
            {"dt": timestamp},
        ).fetchone()

        if exists:
            print(f"Features for {timestamp} already exist.")
            return

        # Read previous AQI
        previous = conn.execute(
            text("""
                SELECT "AQI"
                FROM aqi_features
                ORDER BY datetime DESC
                LIMIT 1
            """)
        ).fetchone()

        if previous is None:
            df["AQI_change_rate"] = 0.0
        else:
            previous_aqi = previous[0]

            if previous_aqi == 0:
                df["AQI_change_rate"] = 0.0
            else:
                df["AQI_change_rate"] = (
                    (df.loc[0, "AQI"] - previous_aqi)
                    / previous_aqi
                )

        # Append new row
        df.to_sql(
            name="aqi_features",
            con=conn,
            if_exists="append",
            index=False
        )

        print(f"Uploaded features for {timestamp}.")


def upload_historical_features():

    df = pd.read_csv(
        PROJECT_ROOT / "Data_collection/data/raw/historical_dataset_hourly.csv"
    )

    # Convert datetime column
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Create feature store table
    df.to_sql(
        name="aqi_features",
        con=engine,
        if_exists="replace",      # Creates/replaces table
        index=False,
        dtype={
            "datetime": DateTime()
        }
    )

    print(f"Uploaded {len(df)} historical feature rows.")

upload_historical_features()
