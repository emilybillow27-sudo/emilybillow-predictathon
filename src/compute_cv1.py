#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import joblib
import pickle

from models import cross_validate_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "trained_models")
TRAINING_CACHE = os.path.join(MODEL_DIR, "training_metadata.pkl")


def main():

    print("\n==============================")
    print("        COMPUTE CV1           ")
    print("==============================\n")

    # Load model + GRM
    model = joblib.load(os.path.join(MODEL_DIR, "final_model.joblib"))
    G = np.load(os.path.join(MODEL_DIR, "GRM.npy"))

    # Load cached metadata
    with open(TRAINING_CACHE, "rb") as f:
        meta = pickle.load(f)

    pheno = meta["pheno_for_gblup"]
    env = meta["env"]
    geno_numeric = meta["geno_numeric"]
    geno_lines = meta["geno_lines_ordered"]

    print("✓ Loaded training phenotype, env, and genotype")

    # Run CV1
    print("\n=== Running CV1 cross-validation ===")
    cv1_results = cross_validate_model(
        train_pheno=pheno,
        geno_numeric=geno_numeric,
        geno_lines=geno_lines,
        env=env,
        G=G,
        n_folds=5
    )

    print("✓ CV1 complete")

    # Compute per-trial accuracy
    print("\n=== Computing per-trial CV1 accuracy ===")
    trial_acc = (
        cv1_results
        .groupby("studyName")
        .apply(lambda df: df["value"].corr(df["pred"]))
        .reset_index()
        .rename(columns={0: "cv1_accuracy"})
    )

    out_path = os.path.join(MODEL_DIR, "cv1_accuracy_by_trial.csv")
    trial_acc.to_csv(out_path, index=False)

    print(f"✓ Saved CV1 accuracy → {out_path}\n")


if __name__ == "__main__":
    main()