#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUBMISSION = f"{REPO_ROOT}/submission"
GENO_ROOT = f"{REPO_ROOT}/data/predictathon"
TRAINING_META = f"{REPO_ROOT}/data/processed/unified_training_pheno_cleaned.csv"

print("======================================")
print(" Validating Predictathon Submission")
print("======================================\n")

# Load training metadata
train = pd.read_csv(TRAINING_META)
training_trials = set(train["trial"].dropna().unique())
training_accessions = set(train["germplasmName_mapped"].dropna().unique())

# Track errors
errors = []

# ---------------------------------------------------------
# Validate each trial folder
# ---------------------------------------------------------
for trial in sorted(os.listdir(SUBMISSION)):
    trial_path = os.path.join(SUBMISSION, trial)
    if not os.path.isdir(trial_path):
        continue

    print(f"Checking {trial}...")

    # Check that trial exists in predictathon data
    geno_dir = os.path.join(GENO_ROOT, trial)
    if not os.path.isdir(geno_dir):
        errors.append(f"{trial}: No matching folder in data/predictathon/")
        continue

    # Load genotyped accessions
    geno_lines_path = os.path.join(geno_dir, "processed", "geno_lines.npy")
    geno_lines = np.load(geno_lines_path, allow_pickle=True).tolist()
    geno_norm = set([str(x).strip() for x in geno_lines])

    for cv_type in ["CV0", "CV00"]:
        cv_dir = os.path.join(trial_path, cv_type)
        if not os.path.isdir(cv_dir):
            errors.append(f"{trial}: Missing {cv_type}/ folder")
            continue

        # Required files
        pred_file = os.path.join(cv_dir, f"{cv_type}_Predictions.csv")
        trials_file = os.path.join(cv_dir, f"{cv_type}_Trials.csv")
        acc_file = os.path.join(cv_dir, f"{cv_type}_Accessions.csv")

        for f in [pred_file, trials_file, acc_file]:
            if not os.path.exists(f):
                errors.append(f"{trial}: Missing file {f}")

        # Validate predictions
        df = pd.read_csv(pred_file)

        # Column check
        if list(df.columns) != ["germplasmName", "prediction"]:
            errors.append(f"{trial}/{cv_type}: Incorrect columns in Predictions.csv")

        # Genotyped-only check
        preds_set = set(df["germplasmName"])
        if not preds_set.issubset(geno_norm):
            errors.append(f"{trial}/{cv_type}: Predictions include ungenotyped accessions")

        # Missing predictions
        if df["prediction"].isna().any():
            errors.append(f"{trial}/{cv_type}: Missing prediction values")

        # Numeric sanity
        if not np.isfinite(df["prediction"]).all():
            errors.append(f"{trial}/{cv_type}: Non-finite prediction values")

        # Validate metadata
        trials_df = pd.read_csv(trials_file)
        acc_df = pd.read_csv(acc_file)

        if set(trials_df["studyName"]) != training_trials:
            errors.append(f"{trial}/{cv_type}: Trials.csv does not match training trials")

        if set(acc_df["germplasmName"]) != training_accessions:
            errors.append(f"{trial}/{cv_type}: Accessions.csv does not match training accessions")

    print(f"  ✓ {trial} passed basic checks\n")

# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------
print("======================================")
if errors:
    print(" Submission has issues:")
    for e in errors:
        print(" -", e)
else:
    print(" All submission files validated successfully!")
print("======================================")