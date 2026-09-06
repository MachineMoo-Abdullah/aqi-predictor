import pandas as pd
from pathlib import Path
import sys
RAW_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(RAW_DIR))

# Read hourly datasets
weather = pd.read_csv(RAW_DIR / "data/raw/weather_hourly.csv")
air = pd.read_csv(RAW_DIR / "data/raw/air_quality_hourly.csv")

# Convert datetime columns
weather["datetime"] = pd.to_datetime(weather["datetime"])
air["datetime"] = pd.to_datetime(air["datetime"])

# Merge on datetime
merged = pd.merge(
    weather,
    air,
    on="datetime",
    how="inner"
)

# Save merged dataset
merged.to_csv(
    RAW_DIR / "historical_dataset_hourly.csv",
    index=False
)

print(merged.head())
print(f"\nRows: {len(merged)}")
print("\nColumns:")
print(merged.columns)