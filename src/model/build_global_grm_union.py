import numpy as np
import pandas as pd
import allel
from pathlib import Path
import argparse


def load_vcf(vcf_path):
    """Load VCF and return samples, dosage matrix, and SNP keys."""
    call = allel.read_vcf(
        str(vcf_path),
        fields=[
            "samples",
            "calldata/GT",
            "variants/CHROM",
            "variants/POS",
            "variants/REF",
            "variants/ALT",
        ],
    )

    samples = call["samples"]
    GT = call["calldata/GT"]  # shape: variants × samples × ploidy

    # Convert GT → dosage (0/1/2)
    dosage = GT.sum(axis=2).astype(float)  # shape: variants × samples

    chrom = call["variants/CHROM"]
    pos = call["variants/POS"]
    ref = call["variants/REF"]
    alt = call["variants/ALT"][:, 0]  # first ALT allele

    # Unique SNP key
    keys = pd.Index(chrom + ":" + pos.astype(str) + ":" + ref + ":" + alt)

    return samples, dosage, keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf_list", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load VCF paths
    vcfs = [Path(p.strip()) for p in open(args.vcf_list)]
    print("Using VCFs:", vcfs)

    # First pass: collect union of SNP keys
    union_keys = set()
    vcf_data = []

    for v in vcfs:
        print(f"Loading {v} ...")
        samples, dosage, keys = load_vcf(v)
        union_keys |= set(keys)
        vcf_data.append((v, samples, dosage, keys))

    union_keys = pd.Index(sorted(list(union_keys)))
    print("Total union SNPs:", len(union_keys))

    # Second pass: align each VCF to union
    all_genotypes = []
    all_samples = []

    for v, samples, dosage, keys in vcf_data:
        print(f"Aligning {v} ...")

        # Build SNPs × samples DataFrame
        df = pd.DataFrame(dosage, index=keys, columns=samples)

        # Reindex to union SNPs
        df = df.reindex(union_keys)

        # Impute missing SNPs with mean dosage
        df = df.fillna(df.mean(axis=1))

        # Store aligned matrix (samples × SNPs)
        all_genotypes.append(df.values.T)
        all_samples.extend(samples)

    # Stack all trials: samples × SNPs
    Gmat = np.vstack(all_genotypes)
    print("Final genotype matrix shape:", Gmat.shape)

        # ---------------------------------------------------------
    # FIX: Impute NaNs in Gmat BEFORE centering/scaling
    # ---------------------------------------------------------
    # Compute column means ignoring NaNs
    col_means = np.nanmean(Gmat, axis=0)

    # Replace NaNs with column means
    inds = np.where(np.isnan(Gmat))
    if inds[0].size > 0:
        Gmat[inds] = np.take(col_means, inds[1])

    # ---------------------------------------------------------
    # Center markers
    # ---------------------------------------------------------
    M = Gmat - Gmat.mean(axis=0, keepdims=True)

    # ---------------------------------------------------------
    # Drop monomorphic markers (std == 0)
    # ---------------------------------------------------------
    std = M.std(axis=0, ddof=1)
    keep = std > 0
    M = M[:, keep]
    std = std[keep]

    print("Markers kept after removing monomorphic sites:", M.shape[1])

    # ---------------------------------------------------------
    # Scale by std
    # ---------------------------------------------------------
    M = M / std

    # ---------------------------------------------------------
    # Compute GRM
    # ---------------------------------------------------------
    GRM = (M @ M.T) / M.shape[1]

    # Save outputs
    np.save(outdir / "G_global_union.npy", Gmat)
    np.save(outdir / "GRM_global_union.npy", GRM)

    with open(outdir / "G_global_union_samples.txt", "w") as f:
        for s in all_samples:
            f.write(s + "\n")

    print("Done. Saved:")
    print(outdir / "G_global_union.npy")
    print(outdir / "GRM_global_union.npy")
    print(outdir / "G_global_union_samples.txt")


if __name__ == "__main__":
    main()