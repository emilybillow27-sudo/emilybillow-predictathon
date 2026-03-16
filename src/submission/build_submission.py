#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------
# Input: your results folders
# ---------------------------------------------------------
CV0_DIR = f"{REPO_ROOT}/results/cv0_predictions"
CV00_DIR = f"{REPO_ROOT}/results/cv00_predictions"

# Training metadata
TRAINING_META = f"{REPO_ROOT}/data/processed/unified_training_pheno_cleaned.csv"

# Genotyped accessions live here
GENO_ROOT = f"{REPO_ROOT}/data/predictathon"

# ---------------------------------------------------------
# Output: final submission folder
# ---------------------------------------------------------
OUTDIR = f"{REPO_ROOT}/submission"
os.makedirs(OUTDIR, exist_ok=True)

print("======================================")
print(" Building Final Predictathon Submission")
print("======================================\n")

# ---------------------------------------------------------
# Load training metadata
# ---------------------------------------------------------
train = pd.read_csv(TRAINING_META)

training_trials = train["trial"].dropna().unique()
training_accessions = train["germplasmName_mapped"].dropna().unique()

def write_metadata(cv_dir, cv_type):
    pd.DataFrame({"studyName": training_trials}).to_csv(
        os.path.join(cv_dir, f"{cv_type}_Trials.csv"), index=False
    )
    pd.DataFrame({"germplasmName": training_accessions}).to_csv(
        os.path.join(cv_dir, f"{cv_type}_Accessions.csv"), index=False
    )

# ---------------------------------------------------------
# Process each trial
# ---------------------------------------------------------
trials = sorted([f.replace(".csv", "") for f in os.listdir(CV0_DIR) if f.endswith(".csv")])

for trial in trials:
    print(f"Processing {trial}...")

    # Load genotyped accessions
    geno_lines_path = f"{GENO_ROOT}/{trial}/processed/geno_lines.npy"
    geno_lines = np.load(geno_lines_path, allow_pickle=True).tolist()
    geno_norm = [str(x).strip() for x in geno_lines]

    # Create trial folder
    trial_out = os.path.join(OUTDIR, trial)
    os.makedirs(trial_out, exist_ok=True)

    # ---------------------------------------------------------
    # CV0
    # ---------------------------------------------------------
    cv0_pred_file = f"{CV0_DIR}/{trial}.csv"
    df0 = pd.read_csv(cv0_pred_file)
    df0 = df0.rename(columns={"line_name": "germplasmName"})
    df0 = df0[df0["germplasmName"].isin(geno_norm)]

    cv0_dir = os.path.join(trial_out, "CV0")
    os.makedirs(cv0_dir, exist_ok=True)
    df0.to_csv(os.path.join(cv0_dir, "CV0_Predictions.csv"), index=False)
    write_metadata(cv0_dir, "CV0")

    # ---------------------------------------------------------
    # CV00
    # ---------------------------------------------------------
    cv00_pred_file = f"{CV00_DIR}/{trial}.csv"
    df00 = pd.read_csv(cv00_pred_file)
    df00 = df00.rename(columns={"line_name": "germplasmName"})
    df00 = df00[df00["germplasmName"].isin(geno_norm)]

    cv00_dir = os.path.join(trial_out, "CV00")
    os.makedirs(cv00_dir, exist_ok=True)
    df00.to_csv(os.path.join(cv00_dir, "CV00_Predictions.csv"), index=False)
    write_metadata(cv00_dir, "CV00")

print(f"\n[submission] Final submission folder built → {OUTDIR}")