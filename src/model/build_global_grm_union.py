#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

def load_trial_genotypes(trial_dir):
    """Load per-trial numeric genotype matrix and SNP list."""

    geno_path = trial_dir / "processed" / "geno_numeric.npy"
    matrix_path = trial_dir / "processed" / "geno_matrix.csv"

    if not geno_path.exists() or not matrix_path.exists():
        raise FileNotFoundError(f"Missing genotype files in {trial_dir}")

    # Load numeric genotype matrix
    G = np.load(geno_path)

    # Load SNP IDs from geno_matrix.csv
    df = pd.read_csv(matrix_path, nrows=0)
    snps = df.columns.tolist()

    # Drop non-SNP columns if present
    non_snp_cols = {"line_name", "germplasmName", "Unnamed: 0"}
    snps = [s for s in snps if s not in non_snp_cols]

    # Ensure SNP count matches G
    if len(snps) != G.shape[1]:
        raise ValueError(
            f"SNP count mismatch in {trial_dir}:\n"
            f"  geno_matrix.csv columns = {len(snps)}\n"
            f"  geno_numeric.npy columns = {G.shape[1]}"
        )

    return G, snps

def align_to_union(G, snps, union_snps):
    """Align a genotype matrix to the global union SNP list."""
    snp_index = {s: i for i, s in enumerate(snps)}
    aligned = np.zeros((G.shape[0], len(union_snps)), dtype=float)

    for j, snp in enumerate(union_snps):
        if snp in snp_index:
            aligned[:, j] = G[:, snp_index[snp]]
        else:
            aligned[:, j] = np.nan  # missing SNP → impute later

    return aligned


def compute_vanraden_grm(G):
    """Compute VanRaden GRM from numeric genotype matrix, robust to all-NaN markers."""
    # Drop markers that are all NaN
    valid_markers = ~np.all(np.isnan(G), axis=0)
    G = G[:, valid_markers]

    # Impute missing values with marker means
    col_means = np.nanmean(G, axis=0)
    inds = np.where(np.isnan(G))
    G[inds] = np.take(col_means, inds[1])

    # Center markers
    M = G - G.mean(axis=0)

    # Allele frequency-based denominator
    p = G.mean(axis=0) / 2.0  # if coded 0/1/2
    denom = 2 * np.sum(p * (1 - p))

    if denom == 0 or np.isnan(denom):
        raise ValueError("Invalid GRM denominator (0 or NaN). Check genotype coding.")

    GRM = (M @ M.T) / denom

    if np.isnan(GRM).any():
        raise ValueError("GRM contains NaNs after computation.")

    return GRM


def main():
    repo = Path(__file__).resolve().parents[2]
    predictathon_dir = repo / "data" / "predictathon"

    trial_dirs = [d for d in predictathon_dir.iterdir() if d.is_dir()]
    print(f"Found {len(trial_dirs)} trials for global GRM construction.")

    # ---------------------------------------------------------
    # Step 1 — Load all genotype matrices + SNP lists
    # ---------------------------------------------------------
    all_G = []
    all_snps = []
    all_lines = []

    for tdir in trial_dirs:
        G, snps = load_trial_genotypes(tdir)
        all_G.append(G)
        all_snps.append(snps)

        # Load line names
        lines = np.load(tdir / "processed" / "geno_lines.npy", allow_pickle=True).tolist()
        all_lines.extend(lines)

    # ---------------------------------------------------------
    # Step 2 — Build global union SNP list
    # ---------------------------------------------------------
    union_snps = sorted(set().union(*all_snps))
    print(f"Global union SNP count: {len(union_snps)}")

    # ---------------------------------------------------------
    # Step 3 — Align each trial to the union SNP list
    # ---------------------------------------------------------
    aligned_mats = []
    for G, snps in zip(all_G, all_snps):
        aligned = align_to_union(G, snps, union_snps)
        aligned_mats.append(aligned)

    # ---------------------------------------------------------
    # Step 4 — Stack into global genotype matrix
    # ---------------------------------------------------------
    G_global = np.vstack(aligned_mats)

    # ---------------------------------------------------------
    # Step 5 — Compute global GRM
    # ---------------------------------------------------------
    print("Computing global GRM...")
    GRM = compute_vanraden_grm(G_global)

    # ---------------------------------------------------------
    # Step 6 — Save outputs
    # ---------------------------------------------------------
    outdir = repo / "data" / "processed" / "global_union"
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "G_global_union.npy", G_global)
    np.save(outdir / "GRM_global_union.npy", GRM)

    with open(outdir / "G_global_union_samples.txt", "w") as f:
        for ln in all_lines:
            f.write(f"{ln}\n")

    print("Global GRM construction complete.")
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()