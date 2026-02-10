import allel
import pandas as pd
import numpy as np
import glob
import os

# Genotype directory
RAW_DIR = "data/raw/genos"

# Find all VCFs
vcf_paths = sorted(
    glob.glob(os.path.join(RAW_DIR, "*.vcf")) +
    glob.glob(os.path.join(RAW_DIR, "*.vcf.gz"))
)

print("Found VCFs:")
for p in vcf_paths:
    print("  ", p)

if len(vcf_paths) == 0:
    raise FileNotFoundError(f"No VCFs found in {RAW_DIR}")

all_geno = []
all_samples = set()

# Read VCFs
for vcf in vcf_paths:
    print(f"\nReading {vcf}")
    callset = allel.read_vcf(vcf, fields=["samples", "calldata/GT", "variants/ID"])

    samples = callset["samples"]
    gt = allel.GenotypeArray(callset["calldata/GT"]).to_n_alt()
    markers = callset["variants/ID"]

    # Add filename prefix to markers
    prefix = os.path.basename(vcf).replace(".vcf", "").replace(".gz", "")
    markers = [f"{prefix}_{m}" for m in markers]

    df = pd.DataFrame(gt.T, columns=markers)
    df.insert(0, "germplasmName", samples)

    all_geno.append(df)
    all_samples.update(samples)

# Combine sample list
all_samples = sorted(list(all_samples))
merged = pd.DataFrame({"germplasmName": all_samples})

# Merge genotype matrices
for df in all_geno:
    merged = merged.merge(df, on="germplasmName", how="left")

# Save output
output_path = "data/processed/geno_merged_raw.csv"
merged.to_csv(output_path, index=False)

print("\n✓ Merged genotype matrix written to:", output_path)
print("Final shape:", merged.shape)
print("Unique accessions:", merged['germplasmName'].nunique())