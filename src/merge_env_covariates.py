#!/usr/bin/env python3
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIST = os.path.join(ROOT, "data", "processed", "env_historical_standardized.csv")
PRED = os.path.join(ROOT, "data", "processed", "env_covariates_standardized.csv")
OUT  = os.path.join(ROOT, "data", "processed", "env_all_standardized.csv")

def main():
    print("=== Loading environment covariates ===")

    hist = pd.read_csv(HIST)
    pred = pd.read_csv(PRED)

    # Ensure studyName exists
    if "studyName" not in hist.columns:
        raise ValueError("Historical env file missing 'studyName'")
    if "studyName" not in pred.columns:
        raise ValueError("Predictathon env file missing 'studyName'")

    # Weather columns expected ONLY in historical envs
    WEATHER_COLS = [
        "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOTCORR", "RH2M", "WS2M",
        "ALLSKY_SFC_SW_DWN"
    ]

    # Validate historical envs have weather
    for c in WEATHER_COLS:
        if c not in hist.columns:
            raise ValueError(f"Historical env file missing column: {c}")

    # Predictathon envs DO NOT need weather columns
    # They only need studyName + whatever metadata you standardized
    pred_cols = ["studyName"] + [c for c in pred.columns if c != "studyName"]

    # Historical envs: keep only weather + studyName
    hist_cols = ["studyName"] + WEATHER_COLS

    # Align columns by union
    all_cols = sorted(set(hist_cols) | set(pred_cols))

    # Reindex both to the same column set
    hist_aligned = hist.reindex(columns=all_cols)
    pred_aligned = pred.reindex(columns=all_cols)

    # Combine
    combined = pd.concat([hist_aligned, pred_aligned], axis=0, ignore_index=True)

    # Drop duplicates just in case
    combined = combined.drop_duplicates(subset=["studyName"])

    combined.to_csv(OUT, index=False)

    print("✓ env_all_standardized.csv created")
    print(combined.head())

if __name__ == "__main__":
    main()