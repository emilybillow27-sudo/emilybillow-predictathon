#!/usr/bin/env python3

import os
import pandas as pd

# Root folder where your predictions live
root = "submission_output"

# All nine Predictathon trials
trials = [
    "AWY1_DVPWA_2024",
    "TCAP_2025_MANKS",
    "25_Big6_SVREC_SVREC",
    "OHRWW_2025_SPO",
    "CornellMaster_2025_McGowan",
    "24Crk_AY2-3",
    "2025_AYT_Aurora",
    "YT_Urb_25",
    "STP1_2025_MCG",
]

summary_rows = []

for trial in trials:

    cv0_path = f"{root}/{trial}/CV0/CV0predictions.csv"
    cv00_path = f"{root}/{trial}/CV00/CV00predictions.csv"

    # Check that both files exist
    if not (os.path.exists(cv0_path) and os.path.exists(cv00_path)):
        summary_rows.append({
            "trial": trial,
            "status": "missing prediction files",
            "mean_abs_diff": None,
            "min_diff": None,
            "max_diff": None,
        })
        continue

    # Load predictions
    cv0 = pd.read_csv(cv0_path)
    cv00 = pd.read_csv(cv00_path)

    # Sort to ensure matching order
    cv0 = cv0.sort_values("germplasmName").reset_index(drop=True)
    cv00 = cv00.sort_values("germplasmName").reset_index(drop=True)

    # Compute absolute differences
    diff = (cv0["pred"] - cv00["pred"]).abs()

    summary_rows.append({
        "trial": trial,
        "status": "OK",
        "mean_abs_diff": diff.mean(),
        "min_diff": diff.min(),
        "max_diff": diff.max(),
    })

# Convert to DataFrame and print
summary = pd.DataFrame(summary_rows)
print("\n=== CV0 vs CV00 Diagnostic Summary ===\n")
print(summary.to_string(index=False))
print("\nDone.\n")