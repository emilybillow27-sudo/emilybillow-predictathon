#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# GRM builder (VanRaden-like)
# ---------------------------------------------------------

def build_grm_from_geno(geno_numeric):
    """
    geno_numeric: numeric-only genotype matrix (rows = lines, cols = markers)
    Accepts either a pandas DataFrame or a NumPy array.
    Returns a NumPy GRM (lines × lines).
    """
    # Convert to NumPy
    if hasattr(geno_numeric, "to_numpy"):
        X = geno_numeric.to_numpy(dtype=float)
    else:
        X = np.asarray(geno_numeric, dtype=float)

    # Impute missing with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])

    # Remove monomorphic markers
    std = X.std(axis=0)
    keep = std > 0
    X = X[:, keep]

    if X.shape[1] == 0:
        raise ValueError("All markers are monomorphic after filtering.")

    # Center markers
    X_centered = X - X.mean(axis=0, keepdims=True)
    m = X_centered.shape[1]

    # VanRaden-like GRM
    G = (X_centered @ X_centered.T) / m

    return G


# ---------------------------------------------------------
# Simple GBLUP (single-environment, per-trial)
# ---------------------------------------------------------

def fit_model(train_pheno, geno_numeric, geno_lines, G, model_type="gblup"):
    """
    train_pheno: DataFrame with ['germplasmName','value'] (single trial)
    geno_numeric: numeric-only genotype matrix (rows = geno_lines)
    geno_lines: list of germplasm names in same order as geno_numeric rows
    G: GRM as a NumPy array with rows/cols in the same order as geno_lines
    """

    df = train_pheno.copy()

    if "germplasmName" not in df.columns or "value" not in df.columns:
        raise ValueError("train_pheno must contain 'germplasmName' and 'value'.")

    # Aggregate to line means (one value per line)
    line_means = (
        df.groupby("germplasmName")["value"]
        .mean()
        .reset_index()
    )

    train_lines = line_means["germplasmName"].tolist()
    y = line_means["value"].astype(float).to_numpy()
    y_mean = y.mean()
    y_centered = y - y_mean

    # Map lines to indices in geno_lines / G
    idx_map = {name: i for i, name in enumerate(geno_lines)}
    train_idx = []
    for ln in train_lines:
        if ln not in idx_map:
            raise ValueError(f"Line '{ln}' not found in geno_lines.")
        train_idx.append(idx_map[ln])
    train_idx = np.array(train_idx, dtype=int)

    # Subset GRM to training lines
    G_train = G[np.ix_(train_idx, train_idx)]

    # Ridge parameter
    lambda_ridge = 1.0
    K = G_train + lambda_ridge * np.eye(G_train.shape[0])

    try:
        u = np.linalg.solve(K, y_centered)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(K, y_centered, rcond=None)[0]

    return {
        "train_lines": train_lines,
        "train_idx": train_idx,
        "u": u,
        "geno_lines": geno_lines,
        "G": G,
        "y_mean": y_mean,
        "lambda_ridge": lambda_ridge,
    }


# ---------------------------------------------------------
# Prediction (per-trial)
# ---------------------------------------------------------

def predict_for_trial(model, focal_trial, test_accessions, geno_numeric, geno_lines, env, G, model_type="gblup"):
    """
    Per-trial prediction using the GRM.
    env and focal_trial are unused here but kept for interface compatibility.
    """

    u = model["u"]
    train_lines = model["train_lines"]
    train_idx = model["train_idx"]
    y_mean = model["y_mean"]
    G_full = model["G"]
    all_lines = model["geno_lines"]

    # Map all lines to indices
    idx_map = {name: i for i, name in enumerate(all_lines)}

    # Build G_test_train: test × train
    G_test_train = []
    for acc in test_accessions:
        if acc in idx_map:
            i = idx_map[acc]
            row = G_full[i, train_idx]
        else:
            row = np.zeros(len(train_idx))
        G_test_train.append(row)
    G_test_train = np.array(G_test_train)

    g_hat = G_test_train @ u
    y_hat = y_mean + g_hat

    return pd.DataFrame({
        "germplasmName": test_accessions,
        "pred": y_hat
    })


# ---------------------------------------------------------
# Optional: simple CV over lines within a trial
# ---------------------------------------------------------

def cross_validate_model(train_pheno, geno_numeric, geno_lines, env, G, model_type="gblup", n_folds=5):
    """
    Simple CV over lines within a single trial.
    Splits lines into folds, trains on (n_folds-1)/n_folds, predicts on held-out lines.
    """

    if "germplasmName" not in train_pheno.columns or "value" not in train_pheno.columns:
        raise ValueError("train_pheno must contain 'germplasmName' and 'value'.")

    # Line means
    line_means = (
        train_pheno.groupby("germplasmName")["value"]
        .mean()
        .reset_index()
    )

    lines = line_means["germplasmName"].tolist()
    rng = np.random.default_rng(42)
    rng.shuffle(lines)
    folds = np.array_split(lines, n_folds)

    results = []

    for fold_id, test_lines in enumerate(folds, start=1):
        test_lines = list(test_lines)
        train_lines = [ln for ln in lines if ln not in test_lines]

        train_df = line_means[line_means["germplasmName"].isin(train_lines)]
        test_df = line_means[line_means["germplasmName"].isin(test_lines)]

        model = fit_model(
            train_pheno=train_df,
            geno_numeric=geno_numeric,
            geno_lines=geno_lines,
            G=G,
            model_type=model_type
        )

        preds = predict_for_trial(
            model=model,
            focal_trial=None,
            test_accessions=test_lines,
            geno_numeric=geno_numeric,
            geno_lines=geno_lines,
            env=None,
            G=G,
            model_type=model_type
        )

        merged = (
            test_df[["germplasmName", "value"]]
            .merge(preds, on="germplasmName", how="left")
        )
        merged["fold"] = fold_id

        results.append(merged)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------
# CLI: build GRM for a trial
# ---------------------------------------------------------

def _cli_build_grm(trial):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geno_path = f"{ROOT}/data/processed/{trial}/geno_matrix.csv"
    outdir = f"{ROOT}/trained_models/{trial}"
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/GRM.npy"

    if not os.path.exists(geno_path):
        raise SystemExit(f"Genotype matrix not found: {geno_path}")

    geno_df = pd.read_csv(geno_path, index_col=0)
    G = build_grm_from_geno(geno_df)

    np.save(outpath, G)
    print(f"[models] {trial}: GRM saved → {outpath}")
    print(f"[models] GRM shape: {G.shape}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "build_grm":
        trial = sys.argv[2]
        _cli_build_grm(trial)
    else:
        raise SystemExit("Usage: python -m src.models build_grm <TRIAL>")


if __name__ == "__main__":
    main()