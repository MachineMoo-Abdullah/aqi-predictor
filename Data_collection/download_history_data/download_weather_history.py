import requests
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import *

START_DATE = "2024-01-01"
END_DATE = "2026-07-30"

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max"
    ],
    "timezone": "auto"
}

response = requests.get(url, params=params)
if response.status_code != 200:
    print(response.text)
    exit()

data = response.json()

df = pd.DataFrame(data["daily"])

RAW_DIR = Path("/Users/altair/Applications/aqi-predictor/Data_collection/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(RAW_DIR / "weather_history.csv", index=False)

print(df.head())
print(f"\nSaved {len(df)} days of weather data.")