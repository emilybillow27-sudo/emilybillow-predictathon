#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

if len(sys.argv) < 2:
    raise SystemExit("Usage: python src/cv00_predict.py <TRIAL>")

trial = sys.argv[1]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output directory
out_dir = f"{ROOT}/results/cv00_predictions"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Load phenotype universe
# ---------------------------------------------------------
pheno = pd.read_csv(
    f"{ROOT}/data/processed/unified_training_pheno_mapped.csv",
    low_memory=False,
)

pheno["value"] = pd.to_numeric(pheno["value"], errors="coerce")

# Unified ID column
pheno["id_for_grm"] = pheno["germplasmName_mapped"].fillna(pheno["germplasmName"])

# ---------------------------------------------------------
# Load GRM
# ---------------------------------------------------------
G = np.load(f"{ROOT}/data/processed/GRM_predictathon.npy")

with open(f"{ROOT}/data/processed/GRM_predictathon_lines.txt") as f:
    grm_lines = [line.strip() for line in f]

line_to_idx = {g: i for i, g in enumerate(grm_lines)}
grm_set = set(grm_lines)

# ---------------------------------------------------------
# Identify focal-trial accessions
# ---------------------------------------------------------
pheno_trial = pheno[pheno["trial"] == trial].copy()

if pheno_trial.empty:
    raise SystemExit(f"No rows found for trial {trial} in phenotype file.")

all_trial_lines = sorted(pheno_trial["id_for_grm"].unique())

genotyped = [g for g in all_trial_lines if g in grm_set]
ungenotyped = [g for g in all_trial_lines if g not in grm_set]

idx_pred = [line_to_idx[g] for g in genotyped]

# ---------------------------------------------------------
# CV00 training set: all other trials WITH phenotypes
# ---------------------------------------------------------
train = pheno[
    (~pheno["id_for_grm"].isin(all_trial_lines)) &
    (~pheno["value"].isna())
].copy()
train = train[train["id_for_grm"].isin(grm_set)]

if train.empty:
    # fallback: training mean = 0 (centered)
    y_pred = np.repeat(0.0, len(all_trial_lines))
    out = pd.DataFrame({"germplasmName": all_trial_lines, "prediction": y_pred})
    out.to_csv(f"{out_dir}/{trial}.csv", index=False)
    print(f"✓ CV00 predictions saved for {trial} (no training data; zero used)")
    sys.exit()

train_lines = sorted(train["id_for_grm"].unique())
idx_train = [line_to_idx[g] for g in train_lines]

# ---------------------------------------------------------
# Build training GRM and phenotype vector
# ---------------------------------------------------------
G_train = G[np.ix_(idx_train, idx_train)]

train_line_means = (
    train.groupby("id_for_grm")["value"]
    .mean()
    .reindex(train_lines)
    .values
)

y = train_line_means
y_mean = y.mean()
y_centered = y - y_mean

# ---------------------------------------------------------
# Solve GBLUP
# ---------------------------------------------------------
lambda_ridge = 1.0
A = G_train + lambda_ridge * np.eye(len(idx_train))
A_jitter = A + 1e-4 * np.eye(A.shape[0])

c, low = cho_factor(A_jitter, check_finite=False)
u_hat = cho_solve((c, low), y_centered, check_finite=False)

# ---------------------------------------------------------
# Predict for genotyped focal-trial accessions
# ---------------------------------------------------------
pred_dict = {}

if genotyped:
    G_pred_train = G[np.ix_(idx_pred, idx_train)]
    y_pred_geno = y_mean + G_pred_train @ u_hat
    for g, val in zip(genotyped, y_pred_geno):
        pred_dict[g] = val

# ---------------------------------------------------------
# Assign training mean to ungenotyped accessions
# ---------------------------------------------------------
for g in ungenotyped:
    pred_dict[g] = y_mean

# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------
out = pd.DataFrame({
    "germplasmName": all_trial_lines,
    "prediction": [pred_dict[g] for g in all_trial_lines]
})

out.to_csv(f"{out_dir}/{trial}.csv", index=False)

print(f"✓ CV00 predictions saved for {trial} (including fallback for ungenotyped lines)")