#!/usr/bin/env python3
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, "data", "processed", "env_covariates.csv")
OUT_PATH = os.path.join(ROOT, "data", "processed", "env_covariates_standardized.csv")

df = pd.read_csv(ENV_PATH)

# Rename your aggregated columns to the names expected by build_env_kernel()
df = df.rename(columns={
    "mean_temp": "T2M",
    "mean_tmax": "T2M_MAX",
    "mean_tmin": "T2M_MIN",
    "total_precip": "PRECTOTCORR",
    "mean_rh": "RH2M",
    "mean_wind": "WS2M",
    "total_rad": "ALLSKY_SFC_SW_DWN",
})

df.to_csv(OUT_PATH, index=False)
print("✓ Standardized environment covariates saved to:", OUT_PATH)
print(df.head())