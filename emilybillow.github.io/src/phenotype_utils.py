# src/phenotype_utils.py

import pandas as pd


# ------------------------------------------------------------
# 1. Clean raw T3 multi-trial phenotype exports
# ------------------------------------------------------------

def clean_raw_t3_file(df):
    """
    Convert a raw wide-format T3 phenotype export into long format
    matching the schema of preprocessed_final.csv.

    Output columns:
        germplasmName, trial, trait_name, value
    """

    # Metadata columns that may exist in T3 exports
    meta_cols = [
        "germplasmName",
        "studyName",
        "studyYear",
        "locationName",
        "replicate",
        "plotNumber",
        "blockNumber",
        "rowNumber",
        "colNumber",
    ]

    # Keep only metadata columns that actually exist
    meta_cols = [c for c in meta_cols if c in df.columns]

    # Trait columns are identified by ontology tags
    trait_cols = [c for c in df.columns if "|CO_" in c]

    # Reshape wide → long
    long = df.melt(
        id_vars=meta_cols,
        value_vars=trait_cols,
        var_name="trait_name",
        value_name="value"
    )

    # Clean trait names
    long["trait_name"] = (
        long["trait_name"]
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Standardize trial name
    long["trial"] = long["studyName"].astype(str).str.strip()

    # Drop missing values
    long = long.dropna(subset=["germplasmName", "trait_name", "value"])

    # Ensure numeric values
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])

    # Final schema
    long = long[["germplasmName", "trial", "trait_name", "value"]]

    return long


# ------------------------------------------------------------
# 2. Standardize yield trait names
# ------------------------------------------------------------

def standardize_yield_traits(df):
    """
    Replace long ontology yield trait names with standardized names.
    """

    mapping = {
        "grain_yield_-_kg/ha|co_321:0001218": "grain_yield_kg_ha",
        "grain_yield_-_g/plot|co_321:0001221": "grain_yield_g_plot",
        "grain_yield_-_main_tillers_-_kg/ha|co_321:0501088": "grain_yield_main_tillers",
    }

    df["trait_name"] = df["trait_name"].map(mapping).fillna(df["trait_name"])
    return df


# ------------------------------------------------------------
# 3. Optional: collapse all yield traits into a single trait
# ------------------------------------------------------------

def collapse_yield_traits(df, target="grain_yield"):
    """
    Collapse all yield traits into a single trait name.
    Useful when building a unified GP dataset.
    """

    yield_traits = [
        "grain_yield_kg_ha",
        "grain_yield_g_plot",
        "grain_yield_main_tillers",
    ]

    df.loc[df["trait_name"].isin(yield_traits), "trait_name"] = target
    return df


# ------------------------------------------------------------
# 4. Existing functions (kept as-is)
# ------------------------------------------------------------

def harmonize_trait_names(pheno_df, vars_df):
    """Replace long trait names with abbreviations when available."""
    mapping = dict(
        zip(vars_df["observationVariableDbId"], vars_df["abbreviation"])
    )

    if "observationVariableDbId" in pheno_df.columns:
        pheno_df["trait"] = pheno_df["observationVariableDbId"].map(mapping).fillna(
            pheno_df.get("trait", pheno_df.get("name"))
        )

    return pheno_df


def extract_environment_covariates(meta_df):
    """Extract environment-level covariates from trial metadata."""
    keep = [
        "studyDbId",
        "location",
        "year",
        "designType",
        "plantingDate",
        "harvestDate",
        "plotWidth",
        "plotLength",
        "fieldSize",
    ]
    return meta_df[keep]