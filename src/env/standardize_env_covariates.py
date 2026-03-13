#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Resolve repo root: src/env → src → repo root
ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
in_path = ROOT / "data/processed/env_historical.csv"
out_path = ROOT / "data/processed/env_historical_standardized.csv"

# Load covariates
df = pd.read_csv(in_path, index_col=0)

# Standardize
df_std = (df - df.mean()) / df.std()

# Save
df_std.to_csv(out_path)

print("Standardized covariates saved.")