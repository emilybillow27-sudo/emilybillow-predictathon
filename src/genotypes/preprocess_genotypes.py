# src/preprocess_genotypes.py

import sys
import os
import numpy as np
import pandas as pd
from cyvcf2 import VCF

def vcf_to_matrix(vcf_path):
    """Convert a filtered VCF to a sample x marker dosage matrix."""
    vcf = VCF(vcf_path)
    samples = vcf.samples

    geno_rows = []
    marker_ids = []

    for variant in vcf:
        g = variant.genotypes  # list of [a1, a2, phased]
        dosages = [a1 + a2 if a1 >= 0 and a2 >= 0 else np.nan for a1, a2, _ in g]
        geno_rows.append(dosages)
        marker_ids.append(f"{variant.CHROM}_{variant.POS}")

    M = pd.DataFrame(np.array(geno_rows).T, index=samples, columns=marker_ids)
    return M


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.preprocess_genotypes <TRIAL>")

    trial = sys.argv[1]

    vcf_path = f"data/predictathon/{trial}/genotypes/{trial}_filtered.vcf.gz"
    if not os.path.exists(vcf_path):
        raise SystemExit(f"Filtered VCF not found: {vcf_path}")

    print(f"[preprocess_genotypes] Loading VCF for {trial}...")
    M = vcf_to_matrix(vcf_path)

    # Output directory
    outdir = f"data/processed/{trial}"
    os.makedirs(outdir, exist_ok=True)

    outpath = f"{outdir}/geno_matrix.csv"
    M.to_csv(outpath)

    print(f"[preprocess_genotypes] {trial}: genotype matrix saved ({M.shape[0]} samples × {M.shape[1]} markers).")


if __name__ == "__main__":
    main()