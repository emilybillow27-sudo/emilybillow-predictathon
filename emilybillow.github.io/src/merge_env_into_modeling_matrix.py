#!/usr/bin/env python3
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pheno_path = os.path.join(ROOT, "data", "processed", "modeling_matrix.csv")
env_path   = os.path.join(ROOT, "data", "processed", "env_all_high_overlap.csv")

out_path   = os.path.join(ROOT, "data", "processed", "modeling_matrix_with_env.csv")

print("Loading phenotype matrix…")
pheno = pd.read_csv(pheno_path)

print("Loading environment covariates…")
env = pd.read_csv(env_path)

# Aggregate env to trial-level means
env_agg = env.groupby("studyName").mean(numeric_only=True).reset_index()

print("Merging phenotype with environment…")
merged = pheno.merge(env_agg, on="studyName", how="left")

print("Saving merged file…")
merged.to_csv(out_path, index=False)

print("✓ modeling_matrix_with_env.csv created.")