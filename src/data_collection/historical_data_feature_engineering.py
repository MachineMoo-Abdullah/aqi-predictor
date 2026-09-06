from pathlib import Path
import pandas as pd
import sys
from .feature_engineering import engineer_historical_features
RAW_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAW_DIR))

raw_path = Path(RAW_DIR/"data/raw/historical_dataset_hourly.csv")
processed_dir = Path(RAW_DIR/"data/processed")
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