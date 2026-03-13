#!/usr/bin/env python3

import os
import pandas as pd

# ---------------------------------------------------------
# Resolve repo root (src/submission → repo root)
# ---------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------
# Input: your existing results
# ---------------------------------------------------------
SOURCE = f"{REPO_ROOT}/submission_output"

# Training metadata (for Trials.csv)
TRAINING_META = f"{REPO_ROOT}/data/processed/unified_training_pheno_cleaned.csv"

# ---------------------------------------------------------
# Output: final submission folder
# ---------------------------------------------------------
OUTDIR = f"{REPO_ROOT}/submission"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------
# Load training metadata
# ---------------------------------------------------------
train = pd.read_csv(TRAINING_META)
training_trials = train["trial"].dropna().unique()

# ---------------------------------------------------------
# Helper to write metadata files
# ---------------------------------------------------------
def write_metadata(cv_dir, cv_type, predicted_accessions):
    # Trials used for training
    pd.DataFrame({"studyName": training_trials}).to_csv(
        os.path.join(cv_dir, f"{cv_type}_Trials.csv"),
        index=False
    )

    # Accessions being predicted for THIS trial
    pd.DataFrame({"germplasmName": predicted_accessions}).to_csv(
        os.path.join(cv_dir, f"{cv_type}_Accessions.csv"),
        index=False
    )

# ---------------------------------------------------------
# Build submission folders
# ---------------------------------------------------------
for trial in os.listdir(SOURCE):
    trial_path = os.path.join(SOURCE, trial)
    if not os.path.isdir(trial_path):
        continue

    # Create trial folder in submission/
    trial_out = os.path.join(OUTDIR, trial)
    os.makedirs(trial_out, exist_ok=True)

    # Process CV0 and CV00
    for cv_type in ["CV0", "CV00"]:
        pred_file = os.path.join(trial_path, f"{cv_type}_Predictions.csv")
        if not os.path.exists(pred_file):
            continue

        # Create CV folder
        cv_dir = os.path.join(trial_out, cv_type)
        os.makedirs(cv_dir, exist_ok=True)

        # Load predictions
        df = pd.read_csv(pred_file)

        # Standardize prediction column
        if "pred_yield" in df.columns:
            df = df.rename(columns={"pred_yield": "prediction"})
        elif "pred" in df.columns:
            df = df.rename(columns={"pred": "prediction"})

        df = df[["germplasmName", "prediction"]]

        # Save predictions
        df.to_csv(os.path.join(cv_dir, f"{cv_type}_Predictions.csv"), index=False)

        # Extract predicted accessions
        predicted_accessions = df["germplasmName"].unique()

        # Write metadata
        write_metadata(cv_dir, cv_type, predicted_accessions)

print(f"[submission] Final submission folder built → {OUTDIR}")