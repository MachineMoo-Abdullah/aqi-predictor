from pathlib import Path
import pandas as pd

from feature_engineering import engineer_historical_features

raw_path = Path("data/raw/historical_dataset_hourly.csv")
processed_dir = Path("data/processed")
processed_path = processed_dir / "historical_dataset_hourly.csv"

processed_dir.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(raw_path)

df = engineer_historical_features(df)


print(df.head())

print(
    "Final shape:",
    df.shape
)

df.to_csv(
    processed_path,
    index=False
)

print(
    f"Saved processed dataset: {processed_path}"
)