#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle

from models import fit_model, predict_for_trial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "trained_models")
TRAINING_CACHE = os.path.join(MODEL_DIR, "training_metadata.pkl")

ENV_FILE = os.path.join(ROOT, "data", "processed", "env_all_standardized.csv")
ACCESSION_DIR = os.path.join(ROOT, "data", "raw", "accession_lists")

SUBMISSION_DIR = os.path.join(ROOT, "predictathon_submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# The 9 Predictathon trials
FOCAL_TRIALS = [
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


def load_training_metadata():
    with open(TRAINING_CACHE, "rb") as f:
        return pickle.load(f)


def load_accessions(trial):
    path = os.path.join(ACCESSION_DIR, f"{trial}.txt")
    with open(path) as f:
        return [x.strip() for x in f]


def write_list(path, values, colname):
    pd.DataFrame({colname: values}).to_csv(path, index=False)


def main():
    print("\n=== Loading metadata ===")
    meta = load_training_metadata()

    pheno = meta["pheno_for_gblup"]
    geno_numeric = meta["geno_numeric"]
    geno_lines = list(meta["geno_lines_ordered"])
    G = np.load(os.path.join(MODEL_DIR, "GRM.npy"))

    print("=== Loading environment covariates ===")
    env = pd.read_csv(ENV_FILE)

    for trial in FOCAL_TRIALS:
        print(f"\n==============================")
        print(f"Processing {trial}")
        print("==============================")

        # Create folder for this trial
        out_dir = os.path.join(SUBMISSION_DIR, trial)
        os.makedirs(out_dir, exist_ok=True)

        # Load focal accessions
        focal_accessions = load_accessions(trial)

        # -------------------------
        # CV0
        # -------------------------
        print("  CV0: building training set...")
        pheno_cv0 = pheno[pheno["studyName"] != trial].copy()

        cv0_trials = sorted(pheno_cv0["studyName"].unique())
        cv0_accessions = sorted(pheno_cv0["germplasmName"].unique())

        print("  CV0: fitting ME-GBLUP...")
        model_cv0 = fit_model(pheno_cv0, geno_numeric, geno_lines, env, G, "me_gblup")

        print("  CV0: predicting...")
        preds_cv0 = predict_for_trial(
            model_cv0, trial, focal_accessions,
            geno_numeric, geno_lines, env, G, "me_gblup"
        )
        preds_cv0.to_csv(os.path.join(out_dir, "CV0_Predictions.csv"), index=False)

        write_list(os.path.join(out_dir, "CV0_Trials.csv"), cv0_trials, "studyName")
        write_list(os.path.join(out_dir, "CV0_Accessions.csv"), cv0_accessions, "germplasmName")

        # -------------------------
        # CV00
        # -------------------------
        print("  CV00: building training set...")
        pheno_cv00 = pheno[pheno["studyName"] != trial].copy()
        pheno_cv00 = pheno_cv00[~pheno_cv00["germplasmName"].isin(focal_accessions)]

        cv00_trials = sorted(pheno_cv00["studyName"].unique())
        cv00_accessions = sorted(pheno_cv00["germplasmName"].unique())

        print("  CV00: fitting ME-GBLUP...")
        model_cv00 = fit_model(pheno_cv00, geno_numeric, geno_lines, env, G, "me_gblup")

        print("  CV00: predicting...")
        preds_cv00 = predict_for_trial(
            model_cv00, trial, focal_accessions,
            geno_numeric, geno_lines, env, G, "me_gblup"
        )
        preds_cv00.to_csv(os.path.join(out_dir, "CV00_Predictions.csv"), index=False)

        write_list(os.path.join(out_dir, "CV00_Trials.csv"), cv00_trials, "studyName")
        write_list(os.path.join(out_dir, "CV00_Accessions.csv"), cv00_accessions, "germplasmName")

    print("\n✓ Predictathon submission folder created at:")
    print(SUBMISSION_DIR)


if __name__ == "__main__":
    main()