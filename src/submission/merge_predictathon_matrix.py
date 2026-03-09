#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

PHENO = Path("data/processed/phenotypes_predictathon.csv")
GENO = Path("data/processed/genotypes_predictathon.csv")
ENV = Path("data/processed/env_covariates_predictathon.csv")
OUT = Path("data/processed/model_matrix_predictathon.csv")

def main():
    # Load phenotype + environment
    pheno = pd.read_csv(PHENO, dtype=str, low_memory=False)
    env = pd.read_csv(ENV, dtype=str, low_memory=False)

    # Merge phenotypes with env covariates
    pheno_env = pheno.merge(env, on="studyName", how="left")

    # Load genotype matrix
    geno = pd.read_csv(GENO, dtype=str, low_memory=False)

    # Reduce genotype matrix to only needed germplasm lines
    keep = pheno_env["germplasmName"].unique()
    geno = geno[geno["germplasmName"].isin(keep)]

    # Convert SNPs to int8 for massive memory savings
    snp_cols = geno.columns[geno.columns.str.startswith("SNP")]
    geno[snp_cols] = geno[snp_cols].astype("int8")

    # Sort to reduce memory fragmentation
    pheno_env = pheno_env.sort_values("germplasmName")
    geno = geno.sort_values("germplasmName")

    # Merge phenotype+env with genotype
    full = pheno_env.merge(geno, on="germplasmName", how="left", sort=False)

    # Report missingness
    missing_env = full["season_length"].isna().sum()
    missing_geno = full[snp_cols].isna().any(axis=1).sum()

    print(f"Rows missing env covariates: {missing_env}")
    print(f"Rows missing genotype data: {missing_geno}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT, index=False)

    print(f"Wrote merged modeling matrix to {OUT}")
    print(f"Final shape: {full.shape}")

if __name__ == "__main__":
    main()