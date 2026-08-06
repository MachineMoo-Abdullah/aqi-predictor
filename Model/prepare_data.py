from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from read_feature import read_features


def prepare_dataset():

    df = read_features()

    # Sort by time
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop("datetime", axis=1)
    # Target
    y = df["AQI"]

    # Use every column except AQI
    X = df.drop(columns=["AQI"])

    return X, y

