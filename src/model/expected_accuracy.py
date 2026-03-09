#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Load data
# -----------------------------
env = pd.read_csv(f"{ROOT}/data/processed/env_covariates_standardized.csv")
pheno = pd.read_csv(f"{ROOT}/data/processed/unified_training_pheno_mapped.csv")

pheno["trial"] = pheno["trial"].astype(str)

# Predictathon trials (from your repo)
PRED_TRIALS = [
    "2025_AYT_Aurora",
    "24Crk_AY2-3",
    "25_Big6_SVREC_SVREC",
    "AWY1_DVPWA_2024",
    "CornellMaster_2025_McGowan",
    "OHRWW_2025_SPO",
    "STP1_2025_MCG",
    "TCAP_2025_MANKS",
    "YT_Urb_25"
]

# -----------------------------
# Helper: get genotype sets
# -----------------------------
def load_genotypes_for_trial(trial):
    path = f"{ROOT}/results/samples/{trial}.txt"
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set([x.strip() for x in f.readlines()])

geno_predictathon = {t: load_genotypes_for_trial(t) for t in PRED_TRIALS}

# Historical trials = all trials in phenotype file
HIST_TRIALS = sorted(pheno["trial"].unique())

# Build genotype sets for historical trials
geno_historical = {
    t: set(pheno.loc[pheno["trial"] == t, "germplasmName_mapped"])
    for t in HIST_TRIALS
}

# -----------------------------
# Climate similarity matrix
# -----------------------------
env_indexed = env.set_index("trial")
env_matrix = env_indexed.values
env_trials = env_indexed.index.tolist()

# Cosine similarity across all trials
clim_sim_matrix = cosine_similarity(env_matrix)

# Convert to DataFrame for easy lookup
clim_sim = pd.DataFrame(
    clim_sim_matrix,
    index=env_trials,
    columns=env_trials
)

# -----------------------------
# Historical signal strength
# -----------------------------
signal_strength = (
    pheno.groupby("trial")["value"]
    .agg(lambda x: np.nanstd(pd.to_numeric(x, errors="coerce")))
)

# -----------------------------
# Expected accuracy calculation
# -----------------------------
rows = []

for p in PRED_TRIALS:
    if p not in clim_sim.index:
        print(f"Warning: no env covariates for {p}")
        continue

    gp = geno_predictathon[p]

    numer = 0.0
    denom = 0.0

    for h in HIST_TRIALS:
        if h not in clim_sim.index:
            continue

        # Climate similarity
        S_clim = clim_sim.loc[p, h]

        # Genotype overlap (Jaccard)
        gh = geno_historical[h]
        if len(gp | gh) == 0:
            S_geno = 0
        else:
            S_geno = len(gp & gh) / len(gp | gh)

        # Historical signal
        A_signal = signal_strength.get(h, np.nan)
        if np.isnan(A_signal):
            continue

        weight = S_clim * S_geno

        numer += weight * A_signal
        denom += weight

    expected_acc = numer / denom if denom > 0 else np.nan

    rows.append({
        "predictathon_trial": p,
        "expected_accuracy": expected_acc
    })

# -----------------------------
# Save output
# -----------------------------
out = pd.DataFrame(rows)
out_path = f"{ROOT}/results/expected_accuracy.csv"
out.to_csv(out_path, index=False)

print(f"Expected accuracy saved → {out_path}")