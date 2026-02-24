#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_LIST = os.path.join(ROOT, "data", "processed", "historical_env_list.csv")
OUT = os.path.join(ROOT, "data", "processed", "historical_env_metadata.csv")

# Default coordinates for major programs
REGION_COORDS = {
    "EWashingtonCereals": (46.73, -117.17),   # Pullman, WA
    "NIdahoCereals": (46.73, -117.00),        # Moscow, ID
    "OregonCereals": (45.52, -122.68),        # Portland, OR (approx)
    "NDK-WHEAT": (46.90, -98.00),             # North Dakota
    "ARS-SRPN": (38.90, -99.32),              # Kansas
    "ARS-NRPN": (46.90, -98.00),              # Northern Plains
    "CornellMaster": (42.45, -76.48),         # Ithaca, NY
    "AYTred": (42.28, -83.74),                # Michigan
    "MSU": (42.73, -84.48),                   # Michigan State
    "Big6": (40.11, -88.24),                  # Illinois
    "OWW": (40.80, -81.94),                   # Ohio
    "OVT": (42.73, -84.48),                   # Michigan
    "AY3-8": (44.97, -93.26),                 # Minnesota
    "TP_": (43.00, -83.00),                   # Michigan/Ohio region
    "23": (40.00, -95.00),                    # Generic fallback for 20xx trials
    "24": (40.00, -95.00),
    "25": (40.00, -95.00),
}

def infer_region(study):
    for key in REGION_COORDS:
        if study.startswith(key):
            return key
    # fallback: use first token before underscore
    return study.split("_")[0]

def infer_year(study):
    # extract first 4-digit year
    for token in study.replace("-", "_").split("_"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None

def infer_dates(year):
    if year is None:
        return ("2020-10-01", "2021-07-15")
    # assume winter wheat unless proven otherwise
    planting = f"{year}-10-01"
    harvest = f"{year+1}-07-15"
    return planting, harvest

def main():
    df = pd.read_csv(ENV_LIST)

    rows = []
    for env in df["studyName"]:
        region = infer_region(env)
        lat, lon = REGION_COORDS.get(region, (40.0, -95.0))

        year = infer_year(env)
        planting, harvest = infer_dates(year)

        rows.append({
            "studyName": env,
            "region": region,
            "lat": lat,
            "lon": lon,
            "planting_date": planting,
            "harvest_date": harvest,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)

    print("✓ historical_env_metadata.csv created")
    print(out_df.head())

if __name__ == "__main__":
    main()