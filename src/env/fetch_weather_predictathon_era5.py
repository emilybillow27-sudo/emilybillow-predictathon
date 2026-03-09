#!/usr/bin/env python3
import pandas as pd
import requests
from pathlib import Path
import time

META = Path("data/processed/metadata_predictathon_geocoded.csv")
OUT = Path("data/processed/env_predictathon_daily.csv")

ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "windspeed_10m_max"
]

def yyyymmdd_to_iso(d):
    d = str(int(d))
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"

def fetch_era5(lat, lon, start, end):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC"
    }
    r = requests.get(ERA5_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    meta = pd.read_csv(META)
    meta = meta.dropna(subset=["plantingDate", "harvestDate"])

    records = []

    for _, row in meta.iterrows():
        study = row["studyName"]
        lat = row["latitude"]
        lon = row["longitude"]

        start_iso = yyyymmdd_to_iso(row["plantingDate"])
        end_iso = yyyymmdd_to_iso(row["harvestDate"])

        print(f"Fetching ERA5 for {study} ({lat}, {lon}) {start_iso}–{end_iso}")

        try:
            data = fetch_era5(lat, lon, start_iso, end_iso)
        except Exception as e:
            print(f"Failed for {study}: {e}")
            continue

        if "daily" not in data:
            print(f"No data returned for {study}")
            continue

        daily = data["daily"]
        dates = daily["time"]

        for i, d in enumerate(dates):
            records.append({
                "studyName": study,
                "date": d.replace("-", ""),  # back to YYYYMMDD
                "tmax": daily["temperature_2m_max"][i],
                "tmin": daily["temperature_2m_min"][i],
                "tmean": daily["temperature_2m_mean"][i],
                "rh2m": daily["relative_humidity_2m_mean"][i],
                "precip": daily["precipitation_sum"][i],
                "srad": daily["shortwave_radiation_sum"][i],
                "wind": daily["windspeed_10m_max"][i],
                "latitude": lat,
                "longitude": lon
            })

        time.sleep(0.5)

    df = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Wrote {len(df)} daily ERA5 rows to {OUT}")

if __name__ == "__main__":
    main()