import requests
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import *

START_DATE = "2024-01-01"
END_DATE = "2026-07-30"

url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi"           
    ],
    "timezone": "auto"
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data["hourly"])

df["time"] = pd.to_datetime(df["time"])

daily_df = (
    df.groupby(df["time"].dt.date)
      .mean(numeric_only=True)
      .reset_index()
)

daily_df.rename(
    columns={"time": "date"},
    inplace=True,
    errors="ignore"
)

RAW_DIR = Path("/Users/altair/Applications/aqi-predictor/Data_collection/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

daily_df.to_csv(
    RAW_DIR / "air_quality_history.csv",
    index=False
)

print(daily_df.head())
print(f"\nSaved {len(daily_df)} days of air quality data.")