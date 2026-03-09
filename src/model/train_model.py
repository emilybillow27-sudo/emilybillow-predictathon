#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import joblib

from src.models import fit_model


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.train_model <TRIAL>")

    trial = sys.argv[1]

    print("\n==============================")
    print(f"   TRAIN MODEL — {trial}")
    print("==============================\n")

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ---------------------------------------------------------
    # Load global modeling matrix
    # ---------------------------------------------------------
    mm_path = f"{ROOT}/data/processed/modeling_matrix.csv"
    if not os.path.exists(mm_path):
        raise SystemExit(f"Modeling matrix not found: {mm_path}")

    mm = pd.read_csv(mm_path)
    print(f"✓ Loaded modeling matrix: {mm.shape[0]} rows")

    # Filter to this trial
    pheno = mm[mm["studyName"] == trial].copy()
    if pheno.empty:
        raise SystemExit(f"No phenotype rows found for trial: {trial}")

    print(f"✓ Trial phenotype rows: {pheno.shape[0]}")

    # ---------------------------------------------------------
    # Load global GRM and subset
    # ---------------------------------------------------------
    grm_path = f"{ROOT}/data/processed/GRM.npy"
    lines_path = f"{ROOT}/data/processed/GRM_lines.txt"

    if not os.path.exists(grm_path):
        raise SystemExit(f"Global GRM not found: {grm_path}")
    if not os.path.exists(lines_path):
        raise SystemExit(f"Global GRM line list not found: {lines_path}")

    G_full = np.load(grm_path)
    lines_full = np.genfromtxt(lines_path, dtype=str, delimiter="\n")

    # Lines in this trial
    trial_lines = pheno["germplasmName"].unique()

    # Subset GRM
    mask = np.isin(lines_full, trial_lines)
    G_sub = G_full[mask][:, mask]

    # Keep line order for model
    geno_lines = lines_full[mask].tolist()

    print(f"✓ Trial GRM: {G_sub.shape}")

    # ---------------------------------------------------------
    # Fit model
    # ---------------------------------------------------------
    print("\n=== Fitting model ===")

    model = fit_model(
        train_pheno=pheno,
        geno_numeric=None,      # no genotype matrix needed
        geno_lines=geno_lines,
        G=G_sub,
        model_type="me_gblup"
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    outdir = f"{ROOT}/trained_models/{trial}"
    os.makedirs(outdir, exist_ok=True)

    joblib.dump(model, f"{outdir}/final_model.joblib")
    np.save(f"{outdir}/GRM.npy", G_sub)

    print(f"✓ Saved model → {outdir}/final_model.joblib")
    print(f"✓ Saved GRM   → {outdir}/GRM.npy")
    print("\n✓ Training complete.\n")


if __name__ == "__main__":
    main()