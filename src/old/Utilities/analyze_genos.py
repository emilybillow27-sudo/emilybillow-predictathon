import allel
import pandas as pd
import glob
import os
import re

# -----------------------------
# CONFIG
# -----------------------------
GENO_DIR = "data/raw/genos"        # folder containing your NEW VCFs
PREDICTATHON_LIST = "data/raw/Accessions.txt"   # your Predictathon list

# -----------------------------
# LOAD PREDICTATHON ACCESSIONS
# -----------------------------
predictathon = pd.read_csv(
    PREDICTATHON_LIST,
    header=None,
    names=["germplasmName"],
    dtype=str
)

predictathon_set = set(predictathon["germplasmName"])
print(f"Loaded {len(predictathon_set)} Predictathon accessions.")

# -----------------------------
# FIND VCF FILES
# -----------------------------
vcf_paths = sorted(
    glob.glob(os.path.join(GENO_DIR, "*.vcf")) +
    glob.glob(os.path.join(GENO_DIR, "*.vcf.gz"))
)

if not vcf_paths:
    raise FileNotFoundError(f"No VCFs found in {GENO_DIR}")

print("\nFound VCFs:")
for p in vcf_paths:
    print("  ", os.path.basename(p))

# -----------------------------
# FUNCTION: detect valid protocol
# -----------------------------
def detect_protocol(vcf_path):
    """
    Reads the header of a VCF and tries to detect:
    - protocol name
    - whether it's a real genotyping protocol
    - whether it's a GRM or metadata file
    """
    protocol = None
    is_valid = False
    is_grm = False

    # GRM files can be detected by filename alone
    if "breedbase_grm" in vcf_path.lower():
        return None, False, True

    with open(vcf_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("##"):
                break

            # Detect Breedbase protocol name
            if "Genotyping protocol name" in line:
                m = re.search(r"=(.*)", line)
                if m:
                    protocol = m.group(1).strip()
                    is_valid = True

            # Detect TASSEL/GBS headers
            if "Tassel" in line or "bcftools" in line:
                is_valid = True
                if protocol is None:
                    protocol = "GBS-like"

    return protocol, is_valid, is_grm

# -----------------------------
# ANALYZE EACH VCF
# -----------------------------
results = []

for vcf in vcf_paths:
    fname = os.path.basename(vcf)
    print(f"\nAnalyzing {fname}")

    protocol, is_valid, is_grm = detect_protocol(vcf)

    if is_grm:
        print("  → Skipping (Breedbase GRM file)")
        continue

    if not is_valid:
        print("  → Skipping (not a valid genotyping protocol)")
        continue

    print(f"  Detected protocol: {protocol}")

    # Extract samples
    try:
        callset = allel.read_vcf(vcf, fields=["samples"])
        samples = set(callset["samples"].astype(str))
    except Exception as e:
        print(f"  ERROR reading samples: {e}")
        continue

    overlap = samples & predictathon_set

    results.append({
        "file": fname,
        "protocol": protocol,
        "num_samples": len(samples),
        "num_overlap": len(overlap),
        "overlap_accessions": sorted(list(overlap))
    })

# -----------------------------
# OUTPUT SUMMARY
# -----------------------------
summary = pd.DataFrame(results)
summary = summary.sort_values("num_overlap", ascending=False)

print("\n================ VALID PROTOCOL SUMMARY ================")
print(summary[["file", "protocol", "num_samples", "num_overlap"]])

summary_path = "data/processed/geno_overlap_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n✓ Saved detailed overlap summary to:", summary_path)