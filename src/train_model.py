#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import joblib

from models import (
    fit_model,
    build_grm_from_geno,
)
from blups import compute_genotype_blups


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

PHENO_PATH = os.path.join(DATA_DIR, "modeling_matrix_with_env.csv")
GENO_PATH  = os.path.join(DATA_DIR, "geno_merged_raw.csv")


def main():

    print("\n==============================")
    print("        TRAINING MODEL        ")
    print("==============================\n")

    # ---------------------------------------------------------
    # Load phenotype + genotype
    # ---------------------------------------------------------
    print("=== Loading processed data ===")
    pheno = pd.read_csv(
        PHENO_PATH,
        dtype={"germplasmName": "category", "studyName": "category"}
    )
    geno = pd.read_csv(GENO_PATH)
    geno.iloc[:, 1:] = geno.iloc[:, 1:].astype("float32")

    print(f"✓ Phenotype rows: {len(pheno)}")
    print(f"✓ Genotype matrix shape: {geno.shape}")

    # ---------------------------------------------------------
    # Filter tiny trials
    # ---------------------------------------------------------
    print("\n=== Filtering trials with < 5 phenotype records ===")
    trial_sizes = pheno.groupby("studyName").size()
    small_trials = trial_sizes[trial_sizes < 5].index.tolist()
    pheno = pheno[~pheno["studyName"].isin(small_trials)].copy()
    pheno["studyName"] = pheno["studyName"].cat.remove_unused_categories()
    print(f"✓ Remaining phenotype rows: {len(pheno)}")

    # ---------------------------------------------------------
    # Extract environment covariates
    # ---------------------------------------------------------
    ENV_COLS = [
        "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOTCORR", "RH2M", "WS2M",
        "ALLSKY_SFC_SW_DWN",
    ]
    env = pheno[["germplasmName", "studyName"] + ENV_COLS].copy()
    print(f"✓ Environment covariates shape: {env.shape}")

    # ---------------------------------------------------------
    # Compute BLUPs
    # ---------------------------------------------------------
    print("\n=== Computing genotype BLUPs ===")
    blups = compute_genotype_blups(pheno)
    print(f"✓ BLUPs for {len(blups)} genotypes")

    pheno_for_gblup = (
        pheno[["germplasmName", "studyName"]]
        .drop_duplicates()
        .merge(blups, on="germplasmName", how="inner")
        .rename(columns={"blup": "value"})
    )

    geno_lines = set(geno["germplasmName"])
    pheno_for_gblup = pheno_for_gblup[
        pheno_for_gblup["germplasmName"].isin(geno_lines)
    ]
    pheno_for_gblup["studyName"] = pheno_for_gblup["studyName"].cat.remove_unused_categories()

    print(f"✓ Phenotype rows after genotype filtering: {len(pheno_for_gblup)}")

    # ---------------------------------------------------------
    # Save training metadata for prediction
    # ---------------------------------------------------------
    print("\n=== Saving training metadata ===")
    pheno_for_gblup.to_csv(os.path.join(MODEL_DIR, "training_pheno_used.csv"), index=False)
    env.to_csv(os.path.join(MODEL_DIR, "training_env_used.csv"), index=False)
    geno.to_csv(os.path.join(MODEL_DIR, "training_geno_used.csv"), index=False)

    # ---------------------------------------------------------
    # Build GRM
    # ---------------------------------------------------------
    print("\n=== Building GRM ===")
    G, geno_lines_ordered = build_grm_from_geno(geno)
    print(f"✓ GRM shape: {G.shape}")

    # ---------------------------------------------------------
    # Fit final reaction-norm GBLUP
    # ---------------------------------------------------------
    print("\n=== Fitting final model ===")
    model = fit_model(
        train_pheno=pheno_for_gblup,
        geno=geno,
        env=env,
        G=G,
        model_type="me_gblup",
    )

    # ---------------------------------------------------------
    # Save model + metadata
    # ---------------------------------------------------------
    print("\n=== Saving trained model ===")

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, "final_model.joblib"),
        compress=0
    )

    G = G.astype("float32")
    np.save(os.path.join(MODEL_DIR, "GRM.npy"), G)

    with open(os.path.join(MODEL_DIR, "GRM_lines.txt"), "w") as f:
        for line in geno_lines_ordered:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()