import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to phenotype file
pheno_path = f"{ROOT}/data/processed/unified_training_pheno_mapped.csv"
pheno = pd.read_csv(pheno_path, low_memory=False)

# Ensure trial column is string
pheno["trial"] = pheno["trial"].astype(str)

# Directory containing accession lists
acc_dir = f"{ROOT}/data/raw/accession_lists"

# List all accession list files
acc_files = [f for f in os.listdir(acc_dir) if f.endswith(".txt")]

placeholder_rows = []

for fname in acc_files:
    trial = fname.replace(".txt", "")  # trial name = filename without extension
    acc_path = os.path.join(acc_dir, fname)

    # Load accessions
    with open(acc_path) as f:
        accessions = [line.strip() for line in f if line.strip()]

    # Check which accessions already exist in phenotype file
    existing = set(pheno.loc[pheno["trial"] == trial, "germplasmName"])

    # Add placeholders for missing accessions
    for acc in accessions:
        if acc not in existing:
            placeholder_rows.append({
                "trial": trial,
                "germplasmName": acc,
                "germplasmName_mapped": acc,
                "value": None
            })

# Append new rows
if placeholder_rows:
    pheno = pd.concat([pheno, pd.DataFrame(placeholder_rows)], ignore_index=True)

# Save updated phenotype file
pheno.to_csv(pheno_path, index=False)

print(f"✓ Added {len(placeholder_rows)} placeholder rows for Predictathon trials.")