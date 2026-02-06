# src/genotype_utils.py

import pandas as pd

def load_genotype_matrix(path):
    """
    Minimal genotype loader.
    Expects a CSV with:
        - accession or germplasmName column
        - marker columns
    """
    print(f"Reading genotype matrix: {path}")
    return pd.read_csv(path)


def merge_pheno_geno(pheno_df, geno_df):
    """
    Merge phenotypes and genotypes on accession/germplasmName.
    Adjust the join key as needed for your dataset.
    """

    # Try common accession column names
    possible_keys = ["germplasmName", "accession", "GID", "genotype"]

    join_key = None
    for key in possible_keys:
        if key in pheno_df.columns and key in geno_df.columns:
            join_key = key
            break

    if join_key is None:
        raise ValueError(
            "Could not find a shared accession column between phenotype and genotype data."
        )

    print(f"Merging on key: {join_key}")

    merged = pheno_df.merge(
        geno_df,
        on=join_key,
        how="left"
    )

    return merged
