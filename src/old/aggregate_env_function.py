import pandas as pd

def aggregate_env(df):
    """
    Aggregate daily NASA POWER data to one row per environment.
    """
    agg = (
        df.groupby("studyName")
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

    return agg