#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "data", "processed", "historical_env_metadata.csv")
OUT = os.path.join(ROOT, "data", "processed", "env_historical_standardized.csv")

# NASA POWER endpoint
BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Variables your model expects
VARS = [
    "T2M", "T2M_MAX", "T2M_MIN",
    "PRECTOTCORR", "RH2M", "WS2M",
    "ALLSKY_SFC_SW_DWN"
]

def fetch_weather(lat, lon, start, end):
    params = {
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": ",".join(VARS),
        "format": "JSON"
    }
    r = requests.get(BASE, params=params)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    return pd.DataFrame(data)

def aggregate_weather(df):
    # df has columns VARS and index = dates
    agg = {
        "T2M": df["T2M"].mean(),
        "T2M_MAX": df["T2M_MAX"].mean(),
        "T2M_MIN": df["T2M_MIN"].mean(),
        "PRECTOTCORR": df["PRECTOTCORR"].sum(),
        "RH2M": df["RH2M"].mean(),
        "WS2M": df["WS2M"].mean(),
        "ALLSKY_SFC_SW_DWN": df["ALLSKY_SFC_SW_DWN"].sum(),
    }
    return agg

def main():
    meta = pd.read_csv(META)

    rows = []
    for _, row in meta.iterrows():
        study = row["studyName"]
        lat = row["lat"]
        lon = row["lon"]
        start = row["planting_date"]
        end = row["harvest_date"]

        print(f"Fetching weather for {study}...")

        try:
            df = fetch_weather(lat, lon, start, end)
            df.index = pd.to_datetime(df.index)
            agg = aggregate_weather(df)
            agg["studyName"] = study
            rows.append(agg)
        except Exception as e:
            print(f"⚠️ Failed for {study}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("✓ env_historical_standardized.csv created")
    print(out.head())

if __name__ == "__main__":
    main()