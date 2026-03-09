import pandas as pd
import requests
import time
from pathlib import Path
import re

INPUT = Path("data/raw/metadata.csv")
OUTPUT = Path("data/processed/metadata_predictathon_geocoded.csv")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

def clean_location(name):
    if pd.isna(name):
        return None

    # Remove leading/trailing quotes
    name = name.strip().strip('"').strip("'")

    # Remove pure numeric junk (e.g., "310", "147")
    if re.fullmatch(r"\d+", name):
        return None

    # Remove trailing commas or stray characters
    name = re.sub(r"[^\w\s,.-]", "", name)

    # Normalize whitespace
    name = " ".join(name.split())

    return name if name else None

def geocode_location(location):
    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": "predictathon-geocoder"})
        r.raise_for_status()
        data = r.json()
        if len(data) == 0:
            return None, None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None, None

def main():
    df = pd.read_csv(INPUT)

    # Keep only needed columns
    df = df[["studyName", "locationName", "plantingDate", "harvestDate"]]

    # Clean location names
    df["locationName_clean"] = df["locationName"].apply(clean_location)

    # Convert dates to YYYYMMDD
    df["plantingDate"] = pd.to_datetime(df["plantingDate"]).dt.strftime("%Y%m%d")
    df["harvestDate"] = pd.to_datetime(df["harvestDate"]).dt.strftime("%Y%m%d")

    # Geocode
    lats, lons = [], []
    for loc in df["locationName_clean"]:
        if loc is None:
            lats.append(None)
            lons.append(None)
            continue
        lat, lon = geocode_location(loc)
        lats.append(lat)
        lons.append(lon)
        time.sleep(1)

    df["latitude"] = lats
    df["longitude"] = lons

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Wrote geocoded metadata to {OUTPUT}")

if __name__ == "__main__":
    main()