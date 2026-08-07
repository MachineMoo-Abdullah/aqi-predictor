import sys
from pathlib import Path

# Project root: .../aqi-predictor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.fetching_live_data.fetch_weather import fetch_hourly_weather
from data_collection.fetching_live_data.fetch_air_quality import fetch_air_quality

from data_collection.feature_engineering import engineer_features

from feature_store.backfill import upload_features


weather, _ = fetch_hourly_weather()

air = fetch_air_quality()

features = engineer_features(weather, air)

upload_features(features)