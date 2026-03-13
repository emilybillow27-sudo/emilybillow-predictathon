#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Input files
HIST = Path("data/processed/env_historical.csv")
PNW = Path("data/processed/env_pnw.csv")

# Output file
OUT = Path("data/processed/env_historical_merged.csv")

# Load both datasets
df_hist = pd.read_csv(HIST)
df_pnw = pd.read_csv(PNW)

# Concatenate
df = pd.concat([df_hist, df_pnw], ignore_index=True)

# Drop duplicates (if any)
df = df.drop_duplicates(subset=["studyName", "date"])

# Sort for reproducibility
df = df.sort_values(["studyName", "date"])

# Write output
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"Merged dataset written to {OUT} with {len(df)} rows.")