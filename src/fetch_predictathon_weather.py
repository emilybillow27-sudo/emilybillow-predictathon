#!/usr/bin/env python3

import os
import pandas as pd
from fetch_historical_weather import fetch_weather  # reuse your existing function

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "data", "raw", "metadata.csv")
OUT = os.path.join(ROOT, "data", "processed", "env_covariates.csv")

FOCAL = [
    "AWY1_DVPWA_2024",
    "TCAP_2025_MANKS",
    "25_Big6_SVREC_SVREC",
    "OHRWW_2025_SPO",
    "CornellMaster_2025_McGowan",
    "24Crk_AY2-3",
    "2025_AYT_Aurora",
    "YT_Urb_25",
    "STP1_2025_MCG",
]

def main():
    print("Fetching Predictathon weather...")

    meta = pd.read_csv(META)
    meta = meta[meta["studyName"].isin(FOCAL)]

    rows = []
    for _, row in meta.iterrows():
        print(f"Fetching weather for {row['studyName']}...")
        df = fetch_weather(row["latitude"], row["longitude"], row["startDate"], row["endDate"])
        df["studyName"] = row["studyName"]
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT, index=False)

    print(f"✓ Predictathon env covariates saved to {OUT}")

if __name__ == "__main__":
    main()