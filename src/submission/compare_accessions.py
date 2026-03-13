#!/usr/bin/env python3

import os
import pandas as pd

# ---------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ACC_DIR = f"{REPO_ROOT}/data/raw/accession_lists"
CV0_DIR = f"{REPO_ROOT}/results/cv0_predictions_yield"
CV00_DIR = f"{REPO_ROOT}/results/cv00_predictions_yield"

# ---------------------------------------------------------
# Helper to load accession list for a trial
# ---------------------------------------------------------
def load_official_accessions(trial):
    path = os.path.join(ACC_DIR, f"{trial}.txt")
    if not os.path.exists(path):
        print(f"[WARN] No accession list found for {trial}")
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

# ---------------------------------------------------------
# Helper to load predicted accessions
# ---------------------------------------------------------
def load_predicted_accessions(pred_file):
    df = pd.read_csv(pred_file)
    return set(df["germplasmName"].astype(str))

# ---------------------------------------------------------
# Compare for both CV0 and CV00
# ---------------------------------------------------------
def compare_dir(pred_dir, label):
    print(f"\n==============================")
    print(f" Comparing {label} predictions")
    print(f"==============================")

    for file in sorted(os.listdir(pred_dir)):
        if not file.endswith(".csv"):
            continue

        trial = file.replace(".csv", "")
        pred_path = os.path.join(pred_dir, file)

        predicted = load_predicted_accessions(pred_path)
        official = load_official_accessions(trial)

        missing = official - predicted
        extra = predicted - official

        print(f"\n--- {trial} ---")
        print(f"Official count:  {len(official)}")
        print(f"Predicted count: {len(predicted)}")

        if not missing and not extra:
            print("✓ PERFECT MATCH")
        else:
            if missing:
                print(f"⚠ Missing {len(missing)} accessions (in official list but not predicted):")
                for m in sorted(list(missing))[:20]:
                    print("   ", m)
                if len(missing) > 20:
                    print("   ...")

            if extra:
                print(f"⚠ Extra {len(extra)} accessions (predicted but not in official list):")
                for e in sorted(list(extra))[:20]:
                    print("   ", e)
                if len(extra) > 20:
                    print("   ...")

# ---------------------------------------------------------
# Run comparisons
# ---------------------------------------------------------
compare_dir(CV0_DIR, "CV0")
compare_dir(CV00_DIR, "CV00")