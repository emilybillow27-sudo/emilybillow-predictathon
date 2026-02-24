#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle
import joblib

from models import fit_model, build_grm_from_geno   # <-- correct import

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "trained_models")
TRAINING_CACHE = os.path.join(MODEL_DIR, "training_metadata.pkl")

IMPUTED_GENO = os.path.join(MODEL_DIR, "geno_numeric_imputed.npy")
IMPUTED_LINES = os.path.join(MODEL_DIR, "geno_lines_imputed.pkl")

OUTPUT_MODEL = os.path.join(MODEL_DIR, "final_model.joblib")
OUTPUT_GRM = os.path.join(MODEL_DIR, "GRM.npy")


def load_genotypes(meta):
    """
    Load imputed genotype matrix if available.
    Otherwise fall back to raw genotypes.
    """

    if os.path.exists(IMPUTED_GENO) and os.path.exists(IMPUTED_LINES):
        print("✓ Using imputed genotype matrix for training")
        geno_numeric = np.load(IMPUTED_GENO)
        with open(IMPUTED_LINES, "rb") as f:
            geno_lines = pickle.load(f)
    else:
        print("⚠ No imputed genotype matrix found — using raw genotypes")
        geno_numeric = meta["geno_numeric"]
        geno_lines = meta["geno_lines_ordered"]

    print(f"Genotype matrix shape: {geno_numeric.shape}")
    return geno_numeric, geno_lines


def main():

    print("\n==============================")
    print("        TRAIN MODEL           ")
    print("==============================\n")

    # ---------------------------------------------------------
    # Load cached metadata
    # ---------------------------------------------------------
    print("=== Loading training metadata ===")
    with open(TRAINING_CACHE, "rb") as f:
        meta = pickle.load(f)

    pheno = meta["pheno_for_gblup"]
    env = meta["env"]

    print(f"✓ Training phenotype rows: {len(pheno)}")

    # ---------------------------------------------------------
    # Load genotype matrix (imputed if available)
    # ---------------------------------------------------------
    geno_numeric, geno_lines = load_genotypes(meta)

    # ---------------------------------------------------------
    # Rebuild GRM using the (possibly imputed) genotype matrix
    # ---------------------------------------------------------
    print("\n=== Computing GRM ===")
    G = build_grm_from_geno(geno_numeric)
    print(f"✓ GRM shape: {G.shape}")

    np.save(OUTPUT_GRM, G)
    print(f"✓ Saved GRM → {OUTPUT_GRM}")

    # ---------------------------------------------------------
    # Fit the multi-environment GBLUP model
    # ---------------------------------------------------------
    print("\n=== Fitting final CV0 model ===")

    model = fit_model(
        train_pheno=pheno,
        geno_numeric=geno_numeric,
        geno_lines=geno_lines,
        env=env,
        G=G,
        model_type="me_gblup",
    )

    joblib.dump(model, OUTPUT_MODEL)
    print(f"✓ Saved final model → {OUTPUT_MODEL}")

    print("\n✓ Training complete.\n")


if __name__ == "__main__":
    main()