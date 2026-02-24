#!/usr/bin/env python3

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_ENV = os.path.join(ROOT, "data", "raw", "metadata.csv")
OUT = os.path.join(ROOT, "data", "processed", "env_covariates.csv")

# Predictathon trials you need env covariates for
FOCAL_TRIALS = [
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
    print("Rebuilding env_covariates.csv from raw metadata...")

    # Load raw metadata (Predictathon env file)
    df = pd.read_csv(RAW_ENV)

    # Filter to Predictathon trials only
    df = df[df["studyName"].isin(FOCAL_TRIALS)]

    if df.empty:
        raise ValueError("No Predictathon trials found in metadata.csv")

    # Save raw covariates
    df.to_csv(OUT, index=False)

    print(f"✓ env_covariates.csv written to {OUT}")
    print(df.head())


if __name__ == "__main__":
    main()