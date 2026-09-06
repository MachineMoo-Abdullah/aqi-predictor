import requests
import pandas as pd
from pathlib import Path
import sys

from ..fetching_live_data.config import *

START_DATE = "2024-01-01"
END_DATE = "2026-09-02"

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

# Keep hourly data
df = pd.DataFrame(data["hourly"])

# Convert timestamp
df["datetime"] = pd.to_datetime(df["time"])

# Drop original time column
df.drop(columns=["time"], inplace=True)

# Rename AQI column
df.rename(columns={"us_aqi": "AQI"}, inplace=True)

# Reorder columns
df = df[
    [
        "datetime",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "AQI"
    ]
]

RAW_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(RAW_DIR))

df.to_csv(
    RAW_DIR / "data/raw/air_quality_hourly.csv",
    index=False
)

print(df.head())
print(f"\nSaved {len(df)} hourly records.")