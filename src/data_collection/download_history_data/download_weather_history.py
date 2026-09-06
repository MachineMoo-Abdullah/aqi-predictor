import requests
import pandas as pd
from pathlib import Path
import sys


from ..fetching_live_data.config import *

START_DATE = "2024-01-01"
END_DATE = "2026-09-02"

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_10m"
    ],
    "timezone": "auto"
}

response = requests.get(url, params=params)

if response.status_code != 200:
    print(response.text)
    exit()

data = response.json()

# Hourly data
df = pd.DataFrame(data["hourly"])

# Convert timestamp
df["datetime"] = pd.to_datetime(df["time"])

# Drop original time column
df.drop(columns=["time"], inplace=True)

RAW_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(RAW_DIR))


df.to_csv(
    RAW_DIR / "data/raw/weather_hourly.csv",
    index=False
)

print(df.head())
print(f"\nSaved {len(df)} hourly weather records.")