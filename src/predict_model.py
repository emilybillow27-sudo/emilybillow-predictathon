#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import joblib

from models import predict_for_trial
from submission import write_submission_files


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "trained_models")
ACCESSION_DIR = os.path.join(ROOT, "data", "raw", "accession_lists")
OUTPUT_ROOT = os.path.join(ROOT, "submission_output")

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
    # Load trained model + metadata
    # ---------------------------------------------------------
    print("=== Loading trained model ===")
    model = joblib.load(os.path.join(MODEL_DIR, "final_model.joblib"))
    G = np.load(os.path.join(MODEL_DIR, "GRM.npy"))

    pheno = pd.read_csv(os.path.join(MODEL_DIR, "training_pheno_used.csv"))
    env   = pd.read_csv(os.path.join(MODEL_DIR, "training_env_used.csv"))
    geno  = pd.read_csv(os.path.join(MODEL_DIR, "training_geno_used.csv"))

    print("✓ Model and data loaded")

    # ---------------------------------------------------------
    # Predict for each challenge trial
    # ---------------------------------------------------------
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for trial in FOCAL_TRIALS:
        acc_file = os.path.join(ACCESSION_DIR, f"{trial}.txt")
        if not os.path.exists(acc_file):
            raise FileNotFoundError(f"Missing accession list: {acc_file}")

        with open(acc_file, "r") as f:
            accessions = [line.strip() for line in f if line.strip()]

        print(f"\n=== Predicting {trial} ({len(accessions)} accessions) ===")

        preds = predict_for_trial(
            model=model,
            focal_trial=trial,
            test_accessions=accessions,
            geno=geno,
            env=env,
            G=G,
            model_type="me_gblup",
        )

        # ---------------------------------------------------------
        # FIXED: write only the trial-specific accessions
        # ---------------------------------------------------------
        for cv_type in ["CV0", "CV00"]:
            write_submission_files(
                trial_name=trial,
                cv_type=cv_type,
                preds_df=preds,
                train_trials=["historical"],
                train_accessions=accessions,   # ⭐ FIXED
                output_root=OUTPUT_ROOT,
            )

    print("\n✓ Prediction + submission generation complete.\n")


if __name__ == "__main__":
    main()