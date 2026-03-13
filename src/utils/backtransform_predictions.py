#!/usr/bin/env python3

import os
import pandas as pd
import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STATS_PATH = f"{REPO_ROOT}/data/processed/trial_yield_stats_raw.csv"

INPUTS = {
    "cv0": f"{REPO_ROOT}/results/cv0_predictions",
    "cv00": f"{REPO_ROOT}/results/cv00_predictions",
}

OUTPUTS = {
    "cv0": f"{REPO_ROOT}/results/cv0_predictions_yield",
    "cv00": f"{REPO_ROOT}/results/cv00_predictions_yield",
}

stats = pd.read_csv(STATS_PATH)

for key in INPUTS:
    in_dir = INPUTS[key]
    out_dir = OUTPUTS[key]
    os.makedirs(out_dir, exist_ok=True)

    for f in glob.glob(f"{in_dir}/*.csv"):
        df = pd.read_csv(f)

        trial = df["trial"].iloc[0]
        s = stats.loc[stats["trial"] == trial]

        if s.empty:
            print(f"[backtransform] No stats for trial {trial}, skipping {f}")
            continue

        mu = s["raw_mean"].iloc[0]
        sd = s["raw_std"].iloc[0]

        df["pred_yield"] = df["pred"] * sd + mu

        out = os.path.join(out_dir, os.path.basename(f))
        df.to_csv(out, index=False)
        print(f"[backtransform] Wrote → {out}")