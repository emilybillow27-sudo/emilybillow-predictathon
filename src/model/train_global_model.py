#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.model.models import fit_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pheno",
        type=str,
        default="data/processed/unified_training_pheno_mapped.csv"
    )
    parser.add_argument(
        "--grm",
        type=str,
        default="data/processed/global_union/GRM_global_union.npy"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="data/processed/global_union/G_global_union_samples.txt"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="trained_models/global_union_model"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load GRM + sample order
    # ---------------------------------------------------------
    print("[train_global_model] Loading GRM...")
    G = np.load(args.grm)

    with open(args.samples) as f:
        grm_samples = [ln.strip() for ln in f]

    print(f"[train_global_model] GRM shape: {G.shape}")
    print(f"[train_global_model] Loaded {len(grm_samples)} sample IDs")

    # ---------------------------------------------------------
    # Load phenotype and aggregate to one value per line
    # ---------------------------------------------------------
    print("[train_global_model] Loading phenotype...")
    ph = pd.read_csv(args.pheno)

    # Aggregate phenotypes by id_for_grm
    ph_agg = (
        ph.groupby("id_for_grm", as_index=False)["value"]
        .mean()
        .rename(columns={"id_for_grm": "germplasmName"})
    )

    # Keep only samples present in GRM
    ph_agg = ph_agg[ph_agg["germplasmName"].isin(grm_samples)]

    # Intersection
    keep = [s for s in grm_samples if s in set(ph_agg["germplasmName"])]

    print(f"[train_global_model] Using {len(keep)} samples with phenotypes")

    # Subset GRM to phenotype samples
    idx = [grm_samples.index(s) for s in keep]
    G_sub = G[np.ix_(idx, idx)]

    # ---------------------------------------------------------
    # Fit global GBLUP using fit_model()
    # ---------------------------------------------------------
    print("[train_global_model] Fitting global GBLUP model...")

    model = fit_model(
        train_pheno=ph_agg,
        geno_numeric=None,
        geno_lines=keep,
        G=G_sub,
        model_type="gblup"
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------
    joblib.dump(model, outdir / "final_model.joblib")

    print(f"[train_global_model] Saved global model → {outdir}")


if __name__ == "__main__":
    main()