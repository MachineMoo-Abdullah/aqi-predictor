import requests
import pandas as pd

from Data_collection.fetching_live_data.config import *


def fetch_air_quality():

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": [
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

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    current = response.json()["current"]

    air = {
        "datetime": current["time"],
        "pm2_5": current["pm2_5"],
        "pm10": current["pm10"],
        "carbon_monoxide": current["carbon_monoxide"],
        "nitrogen_dioxide": current["nitrogen_dioxide"],
        "sulphur_dioxide": current["sulphur_dioxide"],
        "ozone": current["ozone"],
        "AQI": current["us_aqi"]
    }

    return air