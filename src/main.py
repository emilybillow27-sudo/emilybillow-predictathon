#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

from models import (
    fit_model,
    predict_for_trial,
    cross_validate_model,
    build_grm_from_geno,
)
from submission import write_submission_files


# Challenge trials for Predictathon
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

    # --------------------------------------------------------------
    # Resolve repo root so all paths are stable
    # --------------------------------------------------------------
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(ROOT, "data", "processed")
    output_root = os.path.join(ROOT, "submission_output")

    pheno_path = os.path.join(data_dir, "preprocessed_final.csv")
    geno_path = os.path.join(data_dir, "geno_merged_raw.csv")

    # --------------------------------------------------------------
    # Step 1: Load processed data
    # --------------------------------------------------------------
    print("\n=== Loading processed data ===")

    pheno = pd.read_csv(pheno_path)
    geno = pd.read_csv(geno_path)

    print(f"✓ Raw phenotype rows: {len(pheno)}")
    print(f"✓ Genotype matrix shape: {geno.shape}")

    # --------------------------------------------------------------
    # Step 1b: Convert long-format phenotype → modeling-ready format
    # --------------------------------------------------------------
    if {"germplasmName", "value"}.issubset(pheno.columns):
        pheno = (
            pheno.groupby("germplasmName")["value"]
            .mean()
            .reset_index()
        )
        print(f"✓ Collapsed phenotype to {len(pheno)} unique lines")

    # --------------------------------------------------------------
    # Step 1c: Restrict phenotype to lines with genotypes
    # --------------------------------------------------------------
    geno_lines = set(geno["germplasmName"].tolist())
    before = len(pheno)
    pheno = pheno[pheno["germplasmName"].isin(geno_lines)].reset_index(drop=True)
    after = len(pheno)
    print(f"✓ Filtered phenotype to genotyped lines: kept {after}, dropped {before - after}")

    if after == 0:
        raise ValueError("No phenotype lines overlap with genotype lines.")

    # --------------------------------------------------------------
    # Step 2: Build GRM
    # --------------------------------------------------------------
    print("\n=== Building genomic relationship matrix (GRM) ===")
    G, geno_lines_ordered = build_grm_from_geno(geno)
    print(f"✓ GRM shape: {G.shape}")

    # Diagnostic: GRM diagonal range
    print("GRM diag range:",
          float(G.diagonal().min()),
          float(G.diagonal().max()))

    env = None
    MODEL_TYPE = "me_gblup"

    # --------------------------------------------------------------
    # Step 3: Ensure submission folder structure exists
    # --------------------------------------------------------------
    print("\n=== Ensuring submission folder structure ===")

    for trial in FOCAL_TRIALS:
        for cv_type in ["CV0", "CV00"]:
            os.makedirs(os.path.join(output_root, trial, cv_type), exist_ok=True)

    # --------------------------------------------------------------
    # Step 4: Run CV1 cross-validation
    # --------------------------------------------------------------
    print("\n=== Running CV1 cross-validation ===")

    cv_results = cross_validate_model(
        train_pheno=pheno,
        geno=geno,
        env=env,
        G=G,
        model_type=MODEL_TYPE,
        n_folds=5,
    )

    if {"value", "pred"}.issubset(cv_results.columns):
        corr = cv_results["value"].corr(cv_results["pred"])
        print(f"CV1 accuracy (Pearson r): {corr:.3f}")
    else:
        print("Warning: cv_results missing required columns.")

    cv_out = os.path.join(output_root, "cv1_results.csv")
    cv_results.to_csv(cv_out, index=False)
    print(f"✓ Saved CV1 results to {cv_out}")

    # --------------------------------------------------------------
    # Step 5: Fit final model on all training data
    # --------------------------------------------------------------
    print("\n=== Fitting final model on all training data ===")

    model = fit_model(
        train_pheno=pheno,
        geno=geno,
        env=env,
        G=G,
        model_type=MODEL_TYPE,
    )

    # --------------------------------------------------------------
    # Step 6: Predict for challenge trials
    # --------------------------------------------------------------
    print("\n=== Predicting for challenge trials ===")

    all_genotyped = geno["germplasmName"].unique().tolist()

    for trial in FOCAL_TRIALS:
        for cv_type in ["CV0", "CV00"]:
            print(f"\n--- {trial} / {cv_type} ---")
            print(f"  Predicting for {len(all_genotyped)} genotyped accessions.")

            preds_df = predict_for_trial(
                model=model,
                focal_trial=trial,
                test_accessions=all_genotyped,
                geno=geno,
                env=env,
                G=G,
                model_type=MODEL_TYPE,
            )

            write_submission_files(
                trial_name=trial,
                cv_type=cv_type,
                preds_df=preds_df,
                train_trials=["historical"],
                train_accessions=pheno["germplasmName"].unique().tolist(),
                output_root=output_root,
            )

    print("\n✓ Modeling + submission generation complete.\n")


if __name__ == "__main__":
    main()