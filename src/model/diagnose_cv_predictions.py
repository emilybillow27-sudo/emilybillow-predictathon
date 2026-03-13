#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def load_preds(path):
    df = pd.read_csv(path)
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["pred_yield"] = pd.to_numeric(df["pred_yield"], errors="coerce")
    return df

def diagnose_trial(trial_dir):
    cv0_path = trial_dir / "CV0_Predictions.csv"
    cv00_path = trial_dir / "CV00_Predictions.csv"

    if not cv0_path.exists() or not cv00_path.exists():
        return None

    cv0 = load_preds(cv0_path)
    cv00 = load_preds(cv00_path)

    # Align by accession
    merged = cv0.merge(cv00, on="germplasmName", suffixes=("_cv0", "_cv00"))

    # Basic stats
    mean_cv0 = merged["pred_cv0"].mean()
    mean_cv00 = merged["pred_cv00"].mean()
    shrink = mean_cv00 - mean_cv0

    # Rank correlation
    rho, _ = spearmanr(merged["pred_cv0"], merged["pred_cv00"])

    # Zero‑predictions (uninformative)
    pct_zero_cv0 = (merged["pred_cv0"] == 0).mean()
    pct_zero_cv00 = (merged["pred_cv00"] == 0).mean()

    # Yield range
    min_yield = merged["pred_yield_cv0"].min()
    max_yield = merged["pred_yield_cv0"].max()

    # Flags
    flags = []
    if rho < 0.5:
        flags.append("LOW_CORR")
    if pct_zero_cv00 > 0.25:
        flags.append("MANY_ZERO")
    if shrink > -0.05:
        flags.append("LOW_SHRINK")

    return {
        "trial": merged["trial_cv0"].iloc[0],
        "n_accessions": len(merged),
        "mean_cv0": mean_cv0,
        "mean_cv00": mean_cv00,
        "shrink_cv00_minus_cv0": shrink,
        "spearman_rho": rho,
        "pct_zero_cv0": pct_zero_cv0,
        "pct_zero_cv00": pct_zero_cv00,
        "min_pred_yield": min_yield,
        "max_pred_yield": max_yield,
        "flags": ";".join(flags) if flags else ""
    }

def main():
    submission_root = Path("submission_output")
    rows = []

    for trial_dir in submission_root.iterdir():
        if trial_dir.is_dir():
            diag = diagnose_trial(trial_dir)
            if diag:
                rows.append(diag)

    df = pd.DataFrame(rows)
    df = df.sort_values("trial")
    out_path = submission_root / "CV_Diagnostics.csv"
    df.to_csv(out_path, index=False)

    print(f"Diagnostic report written to: {out_path}")
    print(df)

if __name__ == "__main__":
    main()