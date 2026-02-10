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
    # Identify and remove pathological trials (r = -1.0)
    # ---------------------------------------------------------
    cv1_prev_path = os.path.join(output_root, "cv1_results.csv")
    if os.path.exists(cv1_prev_path):
        print("\n=== Filtering pathological trials (r = -1.0) ===")
        cv1_prev = pd.read_csv(cv1_prev_path)

        trial_r = (
            cv1_prev.groupby("studyName")
            .apply(lambda g: np.corrcoef(g["value"], g["pred"])[0, 1])
            .reset_index(name="r")
        )

        bad_trials = trial_r[trial_r["r"] == -1.0]["studyName"].tolist()
        print(f"✓ Removing {len(bad_trials)} pathological trials")

        pheno = pheno[~pheno["studyName"].isin(bad_trials)]
        print(f"✓ Filtered phenotype rows: {len(pheno)}")
    else:
        print("\n(No previous CV1 results found — skipping filtering.)")

    # ---------------------------------------------------------
    # Environment covariates
    # ---------------------------------------------------------
    ENV_COLS = [
        "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOTCORR", "RH2M", "WS2M",
        "ALLSKY_SFC_SW_DWN"
    ]

    missing_env = [c for c in ENV_COLS if c not in pheno.columns]
    if missing_env:
        raise ValueError(f"Missing environment columns: {missing_env}")

    env = (
        pheno[["studyName"] + ENV_COLS]
        .drop_duplicates()
        .copy()
    )
    print(f"✓ Environment table shape: {env.shape}")

    # ---------------------------------------------------------
    # Compute BLUPs (after filtering)
    # ---------------------------------------------------------
    print("\n=== Fitting mixed model for BLUPs ===")
    blups = compute_genotype_blups(pheno)
    print(f"✓ Got BLUPs for {len(blups)} genotypes")

    # ---------------------------------------------------------
    # Build phenotype table for GBLUP
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
    print("GRM diag range:",
          float(G.diagonal().min()),
          float(G.diagonal().max()))

    MODEL_TYPE = "me_gblup"

    # ---------------------------------------------------------
    # Create submission folders
    # ---------------------------------------------------------
    print("\n=== Ensuring submission folder structure ===")
    for trial in FOCAL_TRIALS:
        for cv_type in ["CV0", "CV00"]:
            os.makedirs(os.path.join(output_root, trial, cv_type), exist_ok=True)

    # ---------------------------------------------------------
    # CV1 cross-validation
    # ---------------------------------------------------------
    print("\n=== Running CV1 cross-validation ===")
    cv1_results = cross_validate_model(
        train_pheno=pheno_for_gblup,
        geno=geno,
        env=env,
        G=G,
        model_type=MODEL_TYPE,
        n_folds=5,
    )

    if {"value", "pred"}.issubset(cv1_results.columns):
        corr = cv1_results["value"].corr(cv1_results["pred"])
        print(f"CV1 accuracy (Pearson r): {corr:.3f}")
    else:
        print("Warning: cv_results missing required columns.")

    cv_out = os.path.join(output_root, "cv1_results.csv")
    cv1_results.to_csv(cv_out, index=False)
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

    for trial in FOCAL_TRIALS:
        trial_txt = os.path.join(accession_list_dir, f"{trial}.txt")
        if not os.path.exists(trial_txt):
            raise FileNotFoundError(f"Missing accession list: {trial_txt}")

        with open(trial_txt, "r") as f:
            trial_accessions = [line.strip() for line in f if line.strip()]

        print(f"\nLoaded {len(trial_accessions)} accessions for {trial}")

        for cv_type in ["CV0", "CV00"]:
            print(f"\n--- {trial} / {cv_type} ---")
            print(f"  Predicting for {len(trial_accessions)} accessions")

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