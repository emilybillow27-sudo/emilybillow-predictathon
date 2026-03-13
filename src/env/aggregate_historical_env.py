#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

METADATA = Path("data/processed/historical_env_metadata_completed.csv")
WEATHER_DIR = Path("data/raw/weather_historical")
OUTPUT = Path("data/processed/env_historical.csv")

def main():
    meta = pd.read_csv(METADATA)

    all_rows = []

    for _, row in meta.iterrows():
        study = row["studyName"]
        weather_file = WEATHER_DIR / f"{study}.csv"

        if not weather_file.exists():
            print(f"⚠ Missing weather file for {study}")
            continue

        df = pd.read_csv(weather_file)
        df["studyName"] = study
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No historical weather files found.")

    merged = pd.concat(all_rows, ignore_index=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    print(f"✓ Wrote {len(merged)} rows to {OUTPUT}")

if __name__ == "__main__":
    main()