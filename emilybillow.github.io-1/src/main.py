#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from models import (
    fit_model,
    predict_for_trial,
    cross_validate_model,
    build_grm_from_geno,
)
from submission import write_submission_files
from blups import compute_genotype_blups


# Predictathon challenge trials
FOCAL_TRIALS = [
    "AWY1_DCPWA_2024",
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

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(ROOT, "data", "processed")
    output_root = os.path.join(ROOT, "submission_output")

    pheno_path = os.path.join(data_dir, "modeling_matrix_with_env.csv")
    geno_path = os.path.join(data_dir, "geno_merged_raw.csv")

    # ---------------------------------------------------------
    # Load phenotype + genotype
    # ---------------------------------------------------------
    print("\n=== Loading processed data ===")
    pheno = pd.read_csv(pheno_path)
    geno = pd.read_csv(geno_path)
    print(f"✓ Raw phenotype rows: {len(pheno)}")
    print(f"✓ Genotype matrix shape: {geno.shape}")

    # ---------------------------------------------------------
    # Filter trials with fewer than 5 records
    # ---------------------------------------------------------
    print("\n=== Filtering trials with < 5 phenotype records ===")
    trial_sizes = pheno.groupby("studyName").size()
    small_trials = trial_sizes[trial_sizes < 5].index.tolist()
    print(f"Trials with < 5 records: {len(small_trials)}")

    pheno = pheno[~pheno["studyName"].isin(small_trials)].copy()
    print(f"After removing <5-record trials: {len(pheno)} phenotype rows")

    # ---------------------------------------------------------
    # Extract environment covariates
    # ---------------------------------------------------------
    ENV_COLS = [
        "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOTCORR", "RH2M", "WS2M",
        "ALLSKY_SFC_SW_DWN",
    ]

    missing_env = [c for c in ENV_COLS if c not in pheno.columns]
    if missing_env:
        raise ValueError(
            f"Missing environment columns in modeling_matrix_with_env.csv: {missing_env}"
        )

    env = pheno[["germplasmName", "studyName"] + ENV_COLS].copy()
    print(f"✓ Loaded environment covariates with shape: {env.shape}")

    # ---------------------------------------------------------
    # Compute BLUPs
    # ---------------------------------------------------------
    print("\n=== Fitting mixed model for BLUPs ===")
    blups = compute_genotype_blups(pheno)
    print(f"✓ Got BLUPs for {len(blups)} genotypes")

    # ---------------------------------------------------------
    # Merge BLUPs back with studyName (CRITICAL for G×E)
    # ---------------------------------------------------------
    pheno_for_gblup = (
        pheno[["germplasmName", "studyName"]]
        .drop_duplicates()
        .merge(blups, on="germplasmName", how="inner")
        .rename(columns={"blup": "value"})
    )

    # Keep only genotyped lines
    geno_lines = set(geno["germplasmName"])
    before = len(pheno_for_gblup)
    pheno_for_gblup = pheno_for_gblup[
        pheno_for_gblup["germplasmName"].isin(geno_lines)
    ]
    after = len(pheno_for_gblup)
    print(f"✓ Filtered phenotype to genotyped lines: kept {after}, dropped {before - after}")

    if after == 0:
        raise ValueError("No phenotype lines overlap with genotype lines.")

    # ---------------------------------------------------------
    # Build GRM
    # ---------------------------------------------------------
    print("\n=== Building GRM ===")
    G, geno_lines_ordered = build_grm_from_geno(geno)
    print(f"✓ GRM shape: {G.shape}")
    print("GRM diag range:", float(G.diagonal().min()), float(G.diagonal().max()))

    MODEL_TYPE = "me_gblup"

    # ---------------------------------------------------------
    # Create submission folders
    # ---------------------------------------------------------
    print("\n=== Ensuring submission folder structure ===")
    os.makedirs(output_root, exist_ok=True)
    for trial in FOCAL_TRIALS:
        for cv_type in ["CV0", "CV00"]:
            os.makedirs(os.path.join(output_root, trial, cv_type), exist_ok=True)

    # ---------------------------------------------------------
    # CV1 cross-validation (environment-based)
    # ---------------------------------------------------------
    print("\n=== Running CV1 cross-validation ===")
    cv_results = cross_validate_model(
        train_pheno=pheno_for_gblup,
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

    # ---------------------------------------------------------
    # Fit final model
    # ---------------------------------------------------------
    print("\n=== Fitting final model ===")
    model = fit_model(
        train_pheno=pheno_for_gblup,
        geno=geno,
        env=env,
        G=G,
        model_type=MODEL_TYPE,
    )

    # ---------------------------------------------------------
    # Predict for challenge trials
    # ---------------------------------------------------------
    print("\n=== Predicting for challenge trials ===")

    accession_list_dir = os.path.join(ROOT, "data", "raw", "accession_lists")
    cv_types = ["CV0", "CV00"]

    for trial in FOCAL_TRIALS:
        trial_txt = os.path.join(accession_list_dir, f"{trial}.txt")
        if not os.path.exists(trial_txt):
            raise FileNotFoundError(f"Missing accession list: {trial_txt}")

        with open(trial_txt, "r") as f:
            trial_accessions = [line.strip() for line in f if line.strip()]

        n_acc = len(trial_accessions)
        print(f"\nLoaded {n_acc} accessions for {trial}")

        for cv_type in cv_types:
            print(f"\n--- {trial} / {cv_type} ---")
            print(f"  Predicting for {n_acc} accessions")

            preds_df = predict_for_trial(
                model=model,
                focal_trial=trial,
                test_accessions=trial_accessions,
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
                train_accessions=pheno_for_gblup["germplasmName"].unique().tolist(),
                output_root=output_root,
            )

    print("\n✓ Modeling + submission generation complete.\n")


if __name__ == "__main__":
    main()