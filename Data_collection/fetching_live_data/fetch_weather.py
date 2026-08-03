from config import *
import requests
from datetime import datetime

def fetch_weather():
    """
    Fetch current weather from OpenWeather.
    """

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    weather = {
        "timestamp": datetime.utcfromtimestamp(data["dt"]),

        "city": data["name"],

        "country": data["sys"]["country"],

        "latitude": LATITUDE,

        "longitude": LONGITUDE,

        "temperature": data["main"]["temp"],

        "feels_like": data["main"]["feels_like"],

        "temp_min": data["main"]["temp_min"],

        "temp_max": data["main"]["temp_max"],

        "humidity": data["main"]["humidity"],

        "pressure": data["main"]["pressure"],

        "sea_level": data["main"].get("sea_level"),

        "ground_level": data["main"].get("grnd_level"),

        "visibility": data.get("visibility"),

        "wind_speed": data["wind"]["speed"],

        "wind_direction": data["wind"]["deg"],

        "wind_gust": data["wind"].get("gust"),

        "cloudiness": data["clouds"]["all"],

        "sunrise": datetime.utcfromtimestamp(
            data["sys"]["sunrise"]
        ),

        "sunset": datetime.utcfromtimestamp(
            data["sys"]["sunset"]
        ),

        "weather_main": data["weather"][0]["main"],

        "weather_description": data["weather"][0]["description"]
    }

    return weather, data


