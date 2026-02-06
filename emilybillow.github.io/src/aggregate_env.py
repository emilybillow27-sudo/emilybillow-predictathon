import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(ROOT, "data", "processed", "env_all_predictathon.csv")

env = pd.read_csv(env_path)

# Aggregate to one row per studyName
agg = (
    env.groupby("studyName")
       .agg({
           "T2M": "mean",
           "T2M_MAX": "mean",
           "T2M_MIN": "mean",
           "PRECTOTCORR": "sum",
           "RH2M": "mean",
           "WS2M": "mean",
           "ALLSKY_SFC_SW_DWN": "sum"
       })
       .reset_index()
)

agg = agg.rename(columns={
    "T2M": "mean_temp",
    "T2M_MAX": "mean_tmax",
    "T2M_MIN": "mean_tmin",
    "PRECTOTCORR": "total_precip",
    "RH2M": "mean_rh",
    "WS2M": "mean_wind",
    "ALLSKY_SFC_SW_DWN": "total_rad"
})

out_path = os.path.join(ROOT, "data", "processed", "env_covariates.csv")
agg.to_csv(out_path, index=False)
print(f"✓ Saved aggregated environmental covariates to {out_path}")