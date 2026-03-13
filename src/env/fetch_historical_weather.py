#!/usr/bin/env python3
import csv
import json
import subprocess
from pathlib import Path

# Input metadata file you created
INPUT_FILE = Path("data/processed/pnw_env_metadata.csv")
OUTPUT_FILE = Path("data/processed/env_pnw.csv")

# Absolute path to curl wrapper
CURL_WRAPPER = Path(__file__).resolve().parent / "fetch_weather.sh"

# NASA POWER parameters
PARAMS = ",".join([
    "T2M_MAX",
    "T2M_MIN",
    "T2M",
    "RH2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "WS2M"
])

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def clean_date(d):
    """Convert YYYY-MM-DD → YYYYMMDD."""
    return d.replace("-", "")


def fetch_weather_with_curl(url: str):
    """Call curl wrapper and return parsed JSON or None."""
    try:
        raw = subprocess.check_output(
            [str(CURL_WRAPPER), url],
            stderr=subprocess.STDOUT
        ).decode("utf-8")

        return json.loads(raw)

    except Exception as e:
        print("  curl or JSON error:", e)
        return None


def build_url(lat, lon, start, end):
    """Construct NASA POWER API URL."""
    return (
        f"{BASE_URL}"
        f"?parameters={PARAMS}"
        f"&start={start}"
        f"&end={end}"
        f"&latitude={lat}"
        f"&longitude={lon}"
        f"&community=AG"
        f"&format=JSON"
        f"&product=POWER"
    )


def main():
    if not INPUT_FILE.exists():
        print(f"Missing metadata file: {INPUT_FILE}")
        return

    rows_out = []

    with INPUT_FILE.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            study = row["studyName"]
            lat = row["latitude"]
            lon = row["longitude"]

            start = clean_date(row["plantingDate"])
            end = clean_date(row["harvestDate"])

            print(f"\nFetching weather for {study} ({start} → {end})")

            url = build_url(lat, lon, start, end)
            data = fetch_weather_with_curl(url)

            if not data or "properties" not in data:
                print(f"  Failed to fetch weather for {study}")
                continue

            params = data["properties"]["parameter"]
            dates = list(params["T2M"].keys())

            print(f"  ✓ Retrieved {len(dates)} days")

            for d in dates:
                rows_out.append({
                    "studyName": study,
                    "date": d,
                    "tmax": params["T2M_MAX"][d],
                    "tmin": params["T2M_MIN"][d],
                    "tmean": params["T2M"][d],
                    "rh2m": params["RH2M"][d],
                    "precip": params["PRECTOTCORR"][d],
                    "srad": params["ALLSKY_SFC_SW_DWN"][d],
                    "wind": params["WS2M"][d],
                })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "studyName", "date", "tmax", "tmin", "tmean",
            "rh2m", "precip", "srad", "wind"
        ])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()