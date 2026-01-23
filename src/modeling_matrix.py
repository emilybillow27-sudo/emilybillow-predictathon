#!/usr/bin/env python3
"""
modeling_matrix.py

Build a modeling-ready phenotype matrix from a large CSV file.
"""

import os
import re
import unicodedata
import pandas as pd
from tqdm import tqdm

# Metadata fields to retain
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

# Extra metadata fields
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

# Metadata renames
METADATA_RENAMES = {
    "studyname": "studyName",
    "study_name": "studyName",
    "trial": "studyName",
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

# Trait-name cleaner
def standardize_trait_name(name):
    name = name.lower()
    name = name.replace(" - ", "_").replace(" ", "_").replace("|", "_")
    name = re.sub(r"[()]", "", name)
    name = name.replace(":", "_")
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"__+", "_", name)
    return name.strip("_")

# Missingness (streaming)
def compute_missingness(path, chunksize=200000):
    total_counts = None
    missing_counts = None

    print("\n=== PASS 1: Computing missingness ===")

    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk.columns = [METADATA_RENAMES.get(c.lower(), c) for c in chunk.columns]

        if "trial" in chunk.columns:
            chunk.rename(columns={"trial": "studyName"}, inplace=True)

        if total_counts is None:
            total_counts = chunk.notna().count()
            missing_counts = chunk.isna().sum()
        else:
            total_counts += chunk.notna().count()
            missing_counts += chunk.isna().sum()

    return (missing_counts / total_counts).to_dict()

# Main builder
def build_modeling_matrix(
    path,
    metadata_path,
    chunksize=200000,
    missingness_threshold=0.5,
    standardize_traits=True
):
    metadata_df = pd.read_csv(metadata_path)
    metadata_df.columns = [METADATA_RENAMES.get(c.lower(), c) for c in metadata_df.columns]

    missing = compute_missingness(path, chunksize)

    traits_to_keep = [
        col for col, miss in missing.items()
        if col in METADATA_COLS or miss <= (1 - missingness_threshold)
    ]

    print(f"\nKeeping {len(traits_to_keep)} columns (including metadata)")

    print("\n=== PASS 2: Building modeling-ready matrix ===")
    cleaned_chunks = []

    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk.columns = [METADATA_RENAMES.get(c.lower(), c) for c in chunk.columns]

        if "trial" in chunk.columns:
            chunk.rename(columns={"trial": "studyName"}, inplace=True)

        if "studyName" not in chunk.columns:
            raise KeyError(f"'studyName' missing. Columns: {chunk.columns.tolist()}")

        chunk = chunk[[c for c in chunk.columns if c in traits_to_keep]]

        chunk = chunk.merge(
            metadata_df,
            on="studyName",
            how="left",
            validate="many_to_one"
        )

        if "plantingDate" in chunk.columns:
            chunk["plantingDate"] = pd.to_datetime(chunk["plantingDate"], errors="coerce")
        if "harvestDate" in chunk.columns:
            chunk["harvestDate"] = pd.to_datetime(chunk["harvestDate"], errors="coerce")

        if "plantingDate" in chunk.columns and "harvestDate" in chunk.columns:
            chunk["season_length"] = (chunk["harvestDate"] - chunk["plantingDate"]).dt.days

        if "studyYear" in chunk.columns:
            chunk["studyYear"] = pd.to_numeric(chunk["studyYear"], errors="coerce")

        if standardize_traits:
            new_cols = []
            for c in chunk.columns:
                new_cols.append(c if c in METADATA_COLS else standardize_trait_name(c))
            chunk.columns = new_cols

        cleaned_chunks.append(chunk)

    final_df = pd.concat(cleaned_chunks, ignore_index=True)

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

# Entry point
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