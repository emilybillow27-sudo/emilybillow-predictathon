#!/usr/bin/env python3
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(f"{ROOT}/data/processed/env_covariates.csv", index_col=0)

df_std = (df - df.mean()) / df.std()
df_std.to_csv(f"{ROOT}/data/processed/env_covariates_standardized.csv")

print("Standardized covariates saved.")
