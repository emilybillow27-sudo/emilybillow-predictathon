#!/usr/bin/env python3
import os
import pickle
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "trained_models")
TRAINING_CACHE = os.path.join(MODEL_DIR, "training_metadata.pkl")

OUT = os.path.join(ROOT, "data", "processed", "historical_env_list.csv")

with open(TRAINING_CACHE, "rb") as f:
    meta = pickle.load(f)

pheno = meta["pheno_for_gblup"]

envs = (
    pheno[["studyName"]]
    .drop_duplicates()
    .sort_values("studyName")
)

envs.to_csv(OUT, index=False)
print("✓ historical_env_list.csv created")
print(envs.head())