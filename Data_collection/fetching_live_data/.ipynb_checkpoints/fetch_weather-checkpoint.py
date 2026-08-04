import requests

from Data_collection.fetching_live_data.config import *


def fetch_hourly_weather():
    """
    Fetch current hour weather from Open-Meteo Forecast API.
    Returns features matching the historical dataset.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "wind_speed_10m"
        ],
        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    current = data["current"]

    weather = {

        "datetime": current["time"],

        "temperature_2m": current["temperature_2m"],

        "relative_humidity_2m": current["relative_humidity_2m"],

        "surface_pressure": current["surface_pressure"],

        "precipitation": current["precipitation"],

        "wind_speed_10m": current["wind_speed_10m"]

    }

    return weather, data