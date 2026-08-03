import pandas as pd
from pathlib import Path

RAW_DIR = Path("/Users/altair/Applications/aqi-predictor/Data_collection/data/raw")

weather = pd.read_csv(RAW_DIR / "weather_history.csv")
air = pd.read_csv(RAW_DIR / "air_quality_history.csv")

# Rename if necessary
weather.rename(columns={"time": "date"}, inplace=True)

# Convert to datetime
weather["date"] = pd.to_datetime(weather["date"])
air["date"] = pd.to_datetime(air["date"])

# Merge
merged = pd.merge(
    weather,
    air,
    on="date",
    how="inner"
)

merged.to_csv(
    RAW_DIR / "historical_dataset.csv",
    index=False
)

print(merged.head())
print(f"\nRows: {len(merged)}")