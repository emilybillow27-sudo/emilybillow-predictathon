#!/usr/bin/env python3
"""
modeling_matrix.py

Build a modeling-ready phenotype matrix from a large CSV file.
Now includes:
  - chunk-safe missingness computation
  - trait-name standardization
  - metadata renaming + automatic metadata merge
  - trait filtering by missingness
  - final matrix assembly
  - renaming the single trait column to 'value'

Designed for multi-GB phenotype files.
"""

import os
import re
import unicodedata
import pandas as pd
from tqdm import tqdm


# ============================================================
# Metadata columns that must ALWAYS be preserved
# ============================================================

METADATA_COLS = [
    "studyName",
    "germplasmName",
    "germplasmDbId",
    "programDbId",
    "programname",
    "programdescription",
    "studydbid",
    "studydescription",
    "studydesign",
    "fieldtrialisplannedtobegenotyped",
    "fieldtrialisplannedtocross",
    "plantingdate",
    "harvestdate",
    "locationdbid",
    "studyYear",
    "locationName",
    "observationlevel",
    "observationunitdbid",
    "observationunitname",
    "replicate",
    "block",
    "plotNumber",
    "entryType"
]

# Additional metadata fields from metadata.csv
METADATA_COLS.extend([
    "trialType",
    "breedingProgramName",
    "breedingProgramDescription",
    "breedingProgramDbId",
    "studyDesign",
    "plotWidth",
    "plotLength",
    "plantsPerPlot",
    "numberBlocks",
    "numberReps",
    "managementFactors",
    "fieldSize",
    "folderDbId",
    "folderName",
    "folderDescription",
    "season_length"
])


# Raw → standardized metadata renames
METADATA_RENAMES = {
    "studyname": "studyName",
    "study_name": "studyName",
    "trial": "studyName",          # phenotype file fix
    "trialname": "studyName",
    "trial_name": "studyName",

    "studyyear": "studyYear",
    "locationname": "locationName",
    "germplasmname": "germplasmName",
    "germplasmdbid": "germplasmDbId",
    "programdbid": "programDbId",
    "plotnumber": "plotNumber",
    "entrytype": "entryType",
    "blocknumber": "block",
}


# ============================================================
# Trait-name standardizer
# ============================================================

def standardize_trait_name(name):
    """Convert complex trait names into clean, machine-friendly identifiers."""
    name = name.lower()
    name = name.replace(" - ", "_")
    name = name.replace(" ", "_")
    name = name.replace("|", "_")
    name = re.sub(r"[()]", "", name)
    name = name.replace(":", "_")
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"__+", "_", name)
    name = name.strip("_")
    return name


# ============================================================
# Missingness computation (chunk-safe)
# ============================================================

def compute_missingness(path, chunksize=200000):
    """Compute missingness per column in a streaming, memory-safe way."""
    total_counts = None
    missing_counts = None

    print("\n=== PASS 1: Computing missingness ===")

    for chunk in pd.read_csv(path, chunksize=chunksize):
        # Normalize metadata names early
        chunk.columns = [
            METADATA_RENAMES.get(c.lower(), c) for c in chunk.columns
        ]

        # Fix phenotype 'trial' column
        if "trial" in chunk.columns:
            chunk.rename(columns={"trial": "studyName"}, inplace=True)

        if total_counts is None:
            total_counts = chunk.notna().count()
            missing_counts = chunk.isna().sum()
        else:
            total_counts += chunk.notna().count()
            missing_counts += chunk.isna().sum()

    missing_fraction = (missing_counts / total_counts).to_dict()
    return missing_fraction


# ============================================================
# Modeling matrix builder (orchestrator)
# ============================================================

def build_modeling_matrix(
    path,
    metadata_path,
    chunksize=200000,
    missingness_threshold=0.5,
    standardize_traits=True
):
    """Build a modeling-ready phenotype matrix from a large CSV file."""

    # ------------------------------------------------------------
    # Load metadata once
    # ------------------------------------------------------------
    metadata_df = pd.read_csv(metadata_path)

    # Standardize metadata column names
    metadata_df.columns = [
        METADATA_RENAMES.get(c.lower(), c) for c in metadata_df.columns
    ]

    # ------------------------------------------------------------
    # 1. Compute missingness
    # ------------------------------------------------------------
    missing = compute_missingness(path, chunksize)

    # ------------------------------------------------------------
    # 2. Filter traits by missingness threshold
    # ------------------------------------------------------------
    traits_to_keep = [
        col for col, miss in missing.items()
        if col in METADATA_COLS or miss <= (1 - missingness_threshold)
    ]

    print(f"\nKeeping {len(traits_to_keep)} columns "
          f"(including protected metadata columns)")

    # ------------------------------------------------------------
    # 3. Stream again and build the final matrix
    # ------------------------------------------------------------
    print("\n=== PASS 2: Building modeling-ready matrix ===")

    cleaned_chunks = []

    for chunk in pd.read_csv(path, chunksize=chunksize):

        # Normalize metadata names
        chunk.columns = [
            METADATA_RENAMES.get(c.lower(), c) for c in chunk.columns
        ]

        # Fix phenotype 'trial' column
        if "trial" in chunk.columns:
            chunk.rename(columns={"trial": "studyName"}, inplace=True)

        # Guardrail
        if "studyName" not in chunk.columns:
            raise KeyError(
                f"'studyName' missing in phenotype chunk. Columns: {chunk.columns.tolist()}"
            )

        # Keep only selected columns
        chunk = chunk[[c for c in chunk.columns if c in traits_to_keep]]

        # Merge metadata by studyName
        chunk = chunk.merge(
            metadata_df,
            on="studyName",
            how="left",
            validate="many_to_one"
        )

        # Optional: engineer modeling-ready features
        if "plantingDate" in chunk.columns:
            chunk["plantingDate"] = pd.to_datetime(chunk["plantingDate"], errors="coerce")
        if "harvestDate" in chunk.columns:
            chunk["harvestDate"] = pd.to_datetime(chunk["harvestDate"], errors="coerce")

        if "plantingDate" in chunk.columns and "harvestDate" in chunk.columns:
            chunk["season_length"] = (
                chunk["harvestDate"] - chunk["plantingDate"]
            ).dt.days

        if "studyYear" in chunk.columns:
            chunk["studyYear"] = pd.to_numeric(chunk["studyYear"], errors="coerce")

        # Standardize trait names (NOT metadata)
        if standardize_traits:
            new_cols = []
            for c in chunk.columns:
                if c in METADATA_COLS:
                    new_cols.append(c)
                else:
                    new_cols.append(standardize_trait_name(c))
            chunk.columns = new_cols

        cleaned_chunks.append(chunk)

    # ------------------------------------------------------------
    # 4. Concatenate all chunks
    # ------------------------------------------------------------
    final_df = pd.concat(cleaned_chunks, ignore_index=True)

    # ------------------------------------------------------------
    # 5. Rename the single trait column to 'value'
    # ------------------------------------------------------------
   # Drop trait_name (identifier, not a numeric trait column)
    if "trait_name" in final_df.columns:
        final_df.drop(columns=["trait_name"], inplace=True)

    trait_cols = [c for c in final_df.columns if c not in METADATA_COLS]

    if len(trait_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 trait column, found {len(trait_cols)}: {trait_cols}"
        )

    final_df.rename(columns={trait_cols[0]: "value"}, inplace=True)

    print("\nModeling matrix built successfully.")
    print(f"Final shape: {final_df.shape}")

    return final_df


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    print("\n=== Running modeling_matrix.py as a script ===")

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(ROOT, "data", "processed", "preprocessed_final.csv")
    metadata_path = os.path.join(ROOT, "data", "raw", "metadata.csv")
    output_path = os.path.join(ROOT, "data", "processed", "modeling_matrix.csv")

    print(f"Building modeling matrix from: {input_path}")

    df = build_modeling_matrix(
        path=input_path,
        metadata_path=metadata_path,
        chunksize=200000,
        missingness_threshold=0.5,
        standardize_traits=True
    )

    print(f"\nWriting modeling matrix to: {output_path}")
    df.to_csv(output_path, index=False)

    print(f"\n✓ Done. Final shape: {df.shape}\n")