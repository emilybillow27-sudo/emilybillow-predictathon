#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import joblib
import pickle

from models import fit_model, predict_for_trial
from submission import write_submission_files


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "trained_models")
ACCESSION_DIR = os.path.join(ROOT, "data", "raw", "accession_lists")
OUTPUT_ROOT = os.path.join(ROOT, "submission_output")

TRAINING_CACHE = os.path.join(MODEL_DIR, "training_metadata.pkl")

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


def main():

    print("\n==============================")
    print("        PREDICT MODEL         ")
    print("==============================\n")

    # ---------------------------------------------------------
    # Load trained model + GRM (for CV0)
    # ---------------------------------------------------------
    print("=== Loading trained model ===")
    model_cv0 = joblib.load(os.path.join(MODEL_DIR, "final_model.joblib"))
    G_full = np.load(os.path.join(MODEL_DIR, "GRM.npy"))
    print("✓ Model + GRM loaded")

    # ---------------------------------------------------------
    # Load cached training metadata
    # ---------------------------------------------------------
    print("\n=== Loading cached training metadata ===")
    with open(TRAINING_CACHE, "rb") as f:
        meta = pickle.load(f)

    pheno_full = meta["pheno_for_gblup"]
    env = meta["env"]
    geno_numeric_full = meta["geno_numeric"]
    geno_lines_full = meta["geno_lines_ordered"]

    print("✓ Cached metadata loaded")
    print(f"✓ Training phenotype rows: {len(pheno_full)}")
    print(f"✓ Genotype numeric matrix shape: {geno_numeric_full.shape}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ---------------------------------------------------------
    # Predict for each challenge trial
    # ---------------------------------------------------------
    for trial in FOCAL_TRIALS:

        # Load accessions for this trial
        acc_file = os.path.join(ACCESSION_DIR, f"{trial}.txt")
        if not os.path.exists(acc_file):
            raise FileNotFoundError(f"Missing accession list: {acc_file}")

        with open(acc_file, "r") as f:
            trial_accessions = [line.strip() for line in f if line.strip()]

        print(f"\n=== Predicting {trial} ({len(trial_accessions)} accessions) ===")

        # =========================================================
        # CV0 — use full model, predict only focal-trial accessions
        # =========================================================
        preds_cv0 = predict_for_trial(
            model=model_cv0,
            focal_trial=trial,
            test_accessions=trial_accessions,
            geno_numeric=geno_numeric_full,
            geno_lines=geno_lines_full,
            env=env,
            G=G_full,
            model_type="me_gblup",
        )

        # For the competition, we set the "training accessions" file
        # to the focal-trial accessions (only those being predicted),
        # not all 931 genotyped lines.
        write_submission_files(
            trial_name=trial,
            cv_type="CV0",
            preds_df=preds_cv0,
            train_trials=["historical"],
            train_accessions=trial_accessions,
            output_root=OUTPUT_ROOT,
        )

        # =========================================================
        # CV00 — strict Predictathon definition:
        # remove all focal-trial accessions from the *phenotype*,
        # but still predict for all focal-trial accessions.
        # We keep the full GRM and genotype matrix so that
        # relationships to test lines are available.
        # =========================================================

        # 1. Remove focal-trial accessions from phenotype
        pheno_cv00 = pheno_full[~pheno_full["germplasmName"].isin(trial_accessions)]

        # ---------------------------------------------------------
        # If CV00 training set is empty, fall back to CV0 model
        # ---------------------------------------------------------
        if len(pheno_cv00) == 0:
            print("⚠ CV00 training set is empty — falling back to CV0 model")
            model_cv00 = model_cv0
        else:
            # Refit model using reduced phenotype set,
            # but still with the full genotype matrix and GRM.
            model_cv00 = fit_model(
                train_pheno=pheno_cv00,
                geno_numeric=geno_numeric_full,
                geno_lines=geno_lines_full,
                env=env,
                G=G_full,
                model_type="me_gblup",
            )

        # 2. Predict for the same focal-trial accessions as CV0
        preds_cv00 = predict_for_trial(
            model=model_cv00,
            focal_trial=trial,
            test_accessions=trial_accessions,
            geno_numeric=geno_numeric_full,
            geno_lines=geno_lines_full,
            env=env,
            G=G_full,
            model_type="me_gblup",
        )

        # Again, per your request, CV00_Accessions.csv will list
        # only the focal-trial accessions being predicted.
        write_submission_files(
            trial_name=trial,
            cv_type="CV00",
            preds_df=preds_cv00,
            train_trials=["historical"],
            train_accessions=trial_accessions,
            output_root=OUTPUT_ROOT,
        )

    print("\n✓ Prediction + submission generation complete.\n")


if __name__ == "__main__":
    main()