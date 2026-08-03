import pandas as pd
from connection import engine
df = pd.read_csv(
    "Applications/aqi-predictor/Data_collection/data/raw/historical_dataset.csv"
)

print(df.head())
print(df.shape)

# Upload to Supabase
df.to_sql(
    name="aqi_features",
    con=engine,
    if_exists="replace",   # replace first time
    index=False
)

print("Upload Successful!")