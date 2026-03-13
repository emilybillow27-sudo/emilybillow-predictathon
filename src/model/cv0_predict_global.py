#!/usr/bin/env python3

import argparse
import pathlib
import yaml
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.models import fit_model


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_accession_list(trial, repo_root):
    acc_path = Path(repo_root) / "data" / "raw" / "accession_lists" / f"{trial}.txt"
    if not acc_path.exists():
        raise SystemExit(f"Accession list not found for trial {trial}: {acc_path}")
    with open(acc_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_yield_stats(repo_root):
    stats_path = Path(repo_root) / "data" / "raw" / "global_yield_stats.csv"
    if not stats_path.exists():
        raise SystemExit(f"global_yield_stats.csv not found: {stats_path}")
    stats = pd.read_csv(stats_path)
    mu = float(stats.loc[0, "mean"])
    sd = float(stats.loc[0, "sd"])
    return mu, sd


def load_and_mask_pheno_cv0(unified_pheno_path: Path, trial: str) -> pd.DataFrame:
    pheno = pd.read_csv(unified_pheno_path, low_memory=False)

    # Standardize accession column
    if "germplasm_name" in pheno.columns:
        pheno = pheno.rename(columns={"germplasm_name": "germplasmName"})
    elif "germplasmName" not in pheno.columns:
        raise SystemExit("Unified phenotype file must contain 'germplasm_name' or 'germplasmName'.")

    # Detect trial column
    trial_cols = [c for c in pheno.columns if "trial" in c.lower() or "study" in c.lower()]
    if not trial_cols:
        raise SystemExit("Could not infer trial column in unified phenotype file.")
    trial_col = trial_cols[0]

    # Ensure numeric phenotype
    if "value" not in pheno.columns:
        raise SystemExit("Unified phenotype file must contain a 'value' column.")
    pheno["value"] = pd.to_numeric(pheno["value"], errors="coerce")
    pheno = pheno.dropna(subset=["value"])

    # CV0: exclude focal trial, keep all other phenotypes (including same accessions in other trials)
    pheno_masked = pheno[pheno[trial_col] != trial].copy()

    return pheno_masked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--trial", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    repo_root = ROOT
    global_union_dir = Path(paths["global_union_root"])
    unified_pheno_path = Path(config["phenotypes"]["unified_training"])

    trial = args.trial
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load and mask phenotypes for true CV0
    # ---------------------------------------------------------
    pheno_masked = load_and_mask_pheno_cv0(unified_pheno_path, trial)
    if pheno_masked.empty:
        print(f"[cv0_predict_global] Warning: no training phenotypes remain after CV0 masking for {trial}.")

    # ---------------------------------------------------------
    # Load GLOBAL GRM + sample order + GLOBAL genotype matrix
    # ---------------------------------------------------------
    grm_path = global_union_dir / "GRM_global_union.npy"
    lines_path = global_union_dir / "G_global_union_samples.txt"
    geno_path = global_union_dir / "G_global_union.npy"

    if not grm_path.exists():
        raise SystemExit(f"Global GRM not found: {grm_path}")
    if not lines_path.exists():
        raise SystemExit(f"G_global_union_samples.txt not found: {lines_path}")
    if not geno_path.exists():
        raise SystemExit(f"Global genotype matrix not found: {geno_path}")

    G = np.load(grm_path)
    geno_numeric = np.load(geno_path)

    with open(lines_path, "r") as f:
        grm_lines = [ln.strip() for ln in f if ln.strip()]
    line_to_idx = {ln: i for i, ln in enumerate(grm_lines)}

    # Restrict phenotypes to lines present in the global union
    pheno_masked = pheno_masked[pheno_masked["germplasmName"].isin(grm_lines)].copy()

    # ---------------------------------------------------------
    # Fit model on GLOBAL GRM with CV0-masked phenotypes
    # ---------------------------------------------------------
    print(f"[cv0_predict_global] Fitting CV0 model for {trial} on global union...")
    model = fit_model(
        train_pheno=pheno_masked if not pheno_masked.empty else None,
        geno_numeric=geno_numeric,
        geno_lines=grm_lines,
        G=G,
        model_type="gblup",
    )

    # ---------------------------------------------------------
    # Predict breeding values for ALL GLOBAL GRM lines
    # ---------------------------------------------------------
    if hasattr(model, "predict_from_grm"):
        u_hat = model.predict_from_grm(G)
    else:
        u_hat = model.predict(G)

    # ---------------------------------------------------------
    # Extract predictions for this trial's accessions
    # ---------------------------------------------------------
    test_accessions = load_accession_list(trial, repo_root)

    preds = []
    for acc in test_accessions:
        if acc in line_to_idx:
            preds.append(u_hat[line_to_idx[acc]])
        else:
            preds.append(0.0)

    # ---------------------------------------------------------
    # Back-transform to raw grain yield
    # ---------------------------------------------------------
    mu, sd = load_yield_stats(repo_root)
    pred_yield = mu + sd * np.array(preds)

    df = pd.DataFrame({
        "trial": trial,
        "germplasmName": test_accessions,
        "pred": preds,
        "pred_yield": pred_yield,
    })

    df.to_csv(out_path, index=False)
    print(f"[cv0_predict_global] Saved true CV0 predictions → {out_path}")


if __name__ == "__main__":
    main()