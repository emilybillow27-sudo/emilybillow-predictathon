#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Resolve repo root: src/env → src → repo root
ROOT = Path(__file__).resolve().parents[2]

# Input files
HIST_STD = ROOT / "data/processed/env_historical_standardized.csv"
PREDICT_ENV = ROOT / "data/processed/historical_env_metadata_completed.csv"

# Output file
OUT = ROOT / "data/processed/env_all_standardized.csv"

# Load standardized historical weather
df_hist = pd.read_csv(HIST_STD)

# Load Predictathon environment metadata
df_pred = pd.read_csv(PREDICT_ENV)

# Ensure studyName exists in both
if "studyName" not in df_hist.columns:
    raise ValueError("env_historical_standardized.csv missing 'studyName' column.")

if "studyName" not in df_pred.columns:
    raise ValueError("historical_env_metadata_completed.csv missing 'studyName' column.")

# Merge on studyName
df = df_pred.merge(df_hist, on="studyName", how="left")

# Save output
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"Merged standardized covariates saved to {OUT}")