import requests

from datetime import datetime

from config import *


def fetch_air_quality():
    """
    Fetch current air pollution data.
    """

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": API_KEY
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    pollution = data["list"][0]

    air = {

        "timestamp": datetime.utcfromtimestamp(
            pollution["dt"]
        ),

        "aqi": pollution["main"]["aqi"],

        "co": pollution["components"]["co"],

        "no": pollution["components"]["no"],

        "no2": pollution["components"]["no2"],

        "o3": pollution["components"]["o3"],

        "so2": pollution["components"]["so2"],

        "pm2_5": pollution["components"]["pm2_5"],

        "pm10": pollution["components"]["pm10"],

        "nh3": pollution["components"]["nh3"]
    }

    return air, data
