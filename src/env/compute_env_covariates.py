#!/usr/bin/env python3
import os
import pandas as pd
import xarray as xr
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

weather_dir = f"{ROOT}/data/processed/weather_raw"
metadata = pd.read_csv(f"{ROOT}/data/raw/metadata.csv")

rows = []

for _, row in metadata.iterrows():
    trial = row["studyName"]
    ncfile = f"{weather_dir}/{trial}.nc"

    if not os.path.exists(ncfile):
        print(f"Skipping {trial}: no weather file")
        continue

    ds = xr.open_dataset(ncfile)

    t2m = ds["t2m"] - 273.15
    tmax = ds["mx2t"] - 273.15
    tmin = ds["mn2t"] - 273.15
    precip = ds["tp"] * 1000
    rad = ds["ssrd"] / 3600
    dew = ds["d2m"] - 273.15

    vpd = 0.6108 * np.exp((17.27 * t2m) / (t2m + 237.3)) - \
          0.6108 * np.exp((17.27 * dew) / (dew + 237.3))

    df = pd.DataFrame({
        "tmean": t2m.mean().item(),
        "tmax": tmax.max().item(),
        "tmin": tmin.min().item(),
        "precip_total": precip.sum().item(),
        "rad_total": rad.sum().item(),
        "vpd_mean": vpd.mean().item(),
        "gdd": ((t2m - 10).clip(lower=0)).sum().item(),
        "heat_days": (tmax > 32).sum().item(),
        "cold_days": (tmin < 0).sum().item(),
    }, index=[trial])

    rows.append(df)

out = pd.concat(rows)
out.to_csv(f"{ROOT}/data/processed/env_covariates.csv")
print("Environmental covariates saved.")
