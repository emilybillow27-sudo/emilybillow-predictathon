#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd


# =========================================================
# GRM builder (VanRaden-like)
# =========================================================

def build_grm_from_geno(geno_numeric):
    """
    Build a VanRaden-like genomic relationship matrix (GRM).

    Parameters
    ----------
    geno_numeric : pandas.DataFrame or numpy.ndarray
        Genotype dosage matrix with shape (n_lines, n_markers).
        Values should be 0/1/2 or NaN.

    Returns
    -------
    G : numpy.ndarray
        GRM of shape (n_lines, n_lines).
    """

    # Convert to NumPy
    if hasattr(geno_numeric, "to_numpy"):
        X = geno_numeric.to_numpy(dtype=float)
    else:
        X = np.asarray(geno_numeric, dtype=float)

    n_lines, n_markers = X.shape
    if n_markers == 0:
        raise ValueError("Genotype matrix has zero markers.")

    # Impute missing values with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])

    # Remove monomorphic markers
    std = X.std(axis=0)
    keep = std > 0
    if keep.sum() == 0:
        raise ValueError("All markers are monomorphic after filtering.")
    X = X[:, keep]

    # Center markers
    X_centered = X - X.mean(axis=0, keepdims=True)
    m = X_centered.shape[1]

    # VanRaden-like GRM
    G = (X_centered @ X_centered.T) / m

    return G


# =========================================================
# GBLUP model class
# =========================================================

class GBLUPModel:
    """
    A lightweight, joblib‑serializable GBLUP model wrapper.

    Stores:
      - u_hat: breeding values in the same order as geno_lines
      - lines: list of germplasm names
    """

    def __init__(self, u_hat, lines):
        self.u_hat = np.asarray(u_hat)
        self.lines = list(lines)

    def predict(self, G):
        """
        Standard prediction interface.
        For GBLUP, predictions are simply u_hat for each line.
        """
        return self.u_hat

    def predict_from_grm(self, G):
        """
        Some pipelines call this explicitly.
        We return u_hat regardless of G, since GBLUP predictions
        are already encoded in u_hat.
        """
        return self.u_hat


# =========================================================
# Core fit_model function (clean, unified)
# =========================================================

def fit_model(train_pheno, geno_numeric, geno_lines, G, model_type="gblup"):
    """
    Fit a GBLUP model using the full solution:
        u = G (G_mm + λI)^(-1) y_m
    where:
        - G is the full GRM for all lines
        - G_mm is the submatrix for lines with phenotypes
        - y_m is the phenotype vector for those lines

    Returns a GBLUPModel with u_hat for ALL geno_lines.
    """

    n_lines = len(geno_lines)

    # -----------------------------------------------------
    # Phenotype‑optional mode
    # -----------------------------------------------------
    if train_pheno is None or len(train_pheno) == 0:
        u_hat = np.zeros(n_lines)
        return GBLUPModel(u_hat=u_hat, lines=geno_lines)

    # -----------------------------------------------------
    # Align phenotype with genotype lines
    # -----------------------------------------------------
    ph = train_pheno.copy()
    ph = ph[ph["germplasmName"].isin(geno_lines)]

    pheno_map = dict(zip(ph["germplasmName"], ph["value"]))
    y = np.array([pheno_map.get(line, np.nan) for line in geno_lines])

    mask = ~np.isnan(y)
    if mask.sum() == 0:
        u_hat = np.zeros(n_lines)
        return GBLUPModel(u_hat=u_hat, lines=geno_lines)

    # Lines with phenotypes
    y_m = y[mask]
    G_mm = G[np.ix_(mask, mask)]          # phenotyped × phenotyped
    G_all_m = G[:, mask]                  # all lines × phenotyped

    # -----------------------------------------------------
    # Full GBLUP solution: u = G_all_m (G_mm + λI)^(-1) y_m
    # -----------------------------------------------------
    lam = 1e-5
    A = G_mm + lam * np.eye(G_mm.shape[0])
    alpha = np.linalg.solve(A, y_m)       # (G_mm + λI)^(-1) y_m

    u_hat = G_all_m @ alpha               # breeding values for ALL lines

    return GBLUPModel(u_hat=u_hat, lines=geno_lines)

# =========================================================
# Prediction (kept for compatibility)
# =========================================================

def predict_for_trial(model, focal_trial, test_accessions, geno_numeric, geno_lines, env, G, model_type="gblup"):
    """
    Predict breeding values for test_accessions using the fitted GBLUPModel.
    All lines have non‑zero u_hat from the full GBLUP solution.
    """

    if isinstance(model, GBLUPModel):
        u_hat = model.u_hat
        line_index = {ln: i for i, ln in enumerate(model.lines)}

        preds = []
        for acc in test_accessions:
            idx = line_index.get(acc, None)
            if idx is None:
                preds.append(0.0)
            else:
                preds.append(u_hat[idx])

        return pd.DataFrame({
            "germplasmName": test_accessions,
            "pred": preds
        })

    # Legacy dict path (if still needed)
    if isinstance(model, dict):
        u = model["u"]
        geno_lines_model = model["geno_lines"]
        line_index = {ln: i for i, ln in enumerate(geno_lines_model)}

        preds = []
        for acc in test_accessions:
            idx = line_index.get(acc, None)
            preds.append(u[idx] if idx is not None else 0.0)

        return pd.DataFrame({
            "germplasmName": test_accessions,
            "pred": preds
        })

    raise TypeError("Model must be GBLUPModel or legacy dict model.")

# =========================================================
# Simple CV (unchanged, but now uses GBLUPModel)
# =========================================================

def cross_validate_model(train_pheno, geno_numeric, geno_lines, env, G, model_type="gblup", n_folds=5):
    """
    Simple CV over lines within a single trial.
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


# =========================================================
# CLI: build GRM for a trial
# =========================================================

def _cli_build_grm(trial):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    geno_path = f"{ROOT}/data/predictathon/{trial}/processed/geno_matrix.csv"
    outdir = f"{ROOT}/data/predictathon/{trial}/processed"
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(geno_path):
        raise SystemExit(f"Genotype matrix not found: {geno_path}")

    print(f"[models] Loading genotype matrix for {trial}...")
    geno_df = pd.read_csv(geno_path, index_col=0)

    print(f"[models] Building GRM for {trial}...")
    G = build_grm_from_geno(geno_df)

    outpath = f"{outdir}/GRM.npy"
    np.save(outpath, G)

    print(f"[models] {trial}: GRM saved → {outpath}")
    print(f"[models] GRM shape: {G.shape}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "build_grm":
        trial = sys.argv[2]
        _cli_build_grm(trial)
    else:
        raise SystemExit("Usage: python -m src.model.models build_grm <TRIAL>")


if __name__ == "__main__":
    main()