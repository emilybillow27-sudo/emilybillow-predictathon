#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import joblib

from src.models import predict_for_trial


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.build_predictathon_submission <TRIAL>")

    trial = sys.argv[1]

    print("\n==============================")
    print(f"  BUILD SUBMISSION — {trial}")
    print("==============================\n")

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = f"{ROOT}/trained_models/{trial}/final_model.joblib"
    grm_path   = f"{ROOT}/trained_models/{trial}/GRM.npy"
    geno_path  = f"{ROOT}/data/processed/{trial}/geno_matrix.csv"
    acc_path   = f"{ROOT}/data/predictathon/{trial}/accessions.csv"

    outdir = f"{ROOT}/predictathon_submission/{trial}"
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/submission.csv"

    # ---------------------------------------------------------
    # Load model + GRM
    # ---------------------------------------------------------
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}")
    if not os.path.exists(grm_path):
        raise SystemExit(f"GRM not found: {grm_path}")

    model = joblib.load(model_path)
    G = np.load(grm_path)

    print("✓ Loaded model + GRM")

    # ---------------------------------------------------------
    # Load genotype matrix
    # ---------------------------------------------------------
    if not os.path.exists(geno_path):
        raise SystemExit(f"Genotype matrix not found: {geno_path}")

    geno_df = pd.read_csv(geno_path, index_col=0)
    geno_numeric = geno_df.to_numpy()
    geno_lines = geno_df.index.to_list()

    print(f"✓ Loaded genotype matrix: {geno_numeric.shape}")

    # ---------------------------------------------------------
    # Load accessions to predict
    # ---------------------------------------------------------
    if not os.path.exists(acc_path):
        raise SystemExit(f"Accessions file not found: {acc_path}")

    acc_df = pd.read_csv(acc_path)
    accessions = acc_df["germplasmName"].tolist()

    print(f"✓ Loaded {len(accessions)} accessions")

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------
    print("\n=== Predicting ===")

    pred_df = predict_for_trial(
        model=model,
        focal_trial=trial,
        test_accessions=accessions,
        geno_numeric=geno_numeric,
        geno_lines=geno_lines,
        env=None,
        G=G,
        model_type="gblup"
    )

    # Extract numeric predictions
    preds = pred_df["pred"].values

    # ---------------------------------------------------------
    # Save submission
    # ---------------------------------------------------------
    sub = pd.DataFrame({
        "germplasmName": accessions,
        "predictedValue": preds
    })

    sub.to_csv(outpath, index=False)
    print(f"✓ Saved submission → {outpath}\n")


if __name__ == "__main__":
    main()