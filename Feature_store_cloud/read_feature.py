import pandas as pd
from connection import engine

query = "SELECT * FROM aqi_features"

df = pd.read_sql(query, engine)

print(df.head())