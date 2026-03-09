#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/env_predictathon_daily.csv")
OUTPUT = Path("data/processed/env_covariates_predictathon.csv")

TBASE = 0.0

def saturation_vapor_pressure(T):
    return 0.6108 * np.exp((17.27 * T) / (T + 237.3))

def compute_vpd(tmean, rh):
    es = saturation_vapor_pressure(tmean)
    ea = es * (rh / 100.0)
    return es - ea

def main():
    df = pd.read_csv(INPUT)

    df["vpd"] = compute_vpd(df["tmean"], df["rh2m"])
    df["gdd"] = np.maximum(df["tmean"] - TBASE, 0)

    df["hot_days_30C"] = (df["tmax"] > 30).astype(int)
    df["hot_days_35C"] = (df["tmax"] > 35).astype(int)
    df["cold_days_0C"] = (df["tmin"] < 0).astype(int)

    df["rainy_day"] = (df["precip"] > 1.0).astype(int)
    df["dtr"] = df["tmax"] - df["tmin"]

    agg = df.groupby("studyName").agg(
        season_length=("date", "count"),
        gdd=("gdd", "sum"),
        mean_tmean=("tmean", "mean"),
        mean_tmax=("tmax", "mean"),
        mean_tmin=("tmin", "mean"),
        hot_days_30C=("hot_days_30C", "sum"),
        hot_days_35C=("hot_days_35C", "sum"),
        cold_days_0C=("cold_days_0C", "sum"),
        cum_precip=("precip", "sum"),
        rainy_days=("rainy_day", "sum"),
        mean_rh=("rh2m", "mean"),
        mean_vpd=("vpd", "mean"),
        cum_vpd=("vpd", "sum"),
        cum_srad=("srad", "sum"),
        mean_srad=("srad", "mean"),
        mean_wind=("wind", "mean"),
        mean_dtr=("dtr", "mean")
    ).reset_index()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUTPUT, index=False)

    print(f"Wrote {len(agg)} rows to {OUTPUT}")

if __name__ == "__main__":
    main()