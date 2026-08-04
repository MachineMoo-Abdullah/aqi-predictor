import requests
import pandas as pd

from Data_collection.fetching_live_data.config import *

def fetch_forecast():

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    forecasts = []

    for item in data["list"]:

        forecasts.append({

            "datetime": item["dt_txt"],

            "temperature": item["main"]["temp"],

            "feels_like": item["main"]["feels_like"],

            "temp_min": item["main"]["temp_min"],

            "temp_max": item["main"]["temp_max"],

            "humidity": item["main"]["humidity"],

            "pressure": item["main"]["pressure"],

            "clouds": item["clouds"]["all"],

            "wind_speed": item["wind"]["speed"],

            "wind_direction": item["wind"]["deg"],

            "weather": item["weather"][0]["main"]

        })

    return pd.DataFrame(forecasts), data

