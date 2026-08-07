import os
from pathlib import Path
from dotenv import load_dotenv

# Get the directory containing config.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load the .env file explicitly
dotenv_path = BASE_DIR.parent / ".env"
load_dotenv(dotenv_path)

# Read variables
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LATITUDE = float(os.getenv("LATITUDE"))
LONGITUDE = float(os.getenv("LONGITUDE"))
CITY = os.getenv("CITY")
COUNTRY = os.getenv("COUNTRY")

# API URLs
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
AIR_QUALITY_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Data folders
RAW_FOLDER = BASE_DIR / "data" / "raw"
PROCESSED_FOLDER = BASE_DIR / "data" / "processed"

# Create folders if they don't exist
RAW_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)