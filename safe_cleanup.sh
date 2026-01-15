#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " SAFE REPO CLEANUP (dry-run first)"
echo "======================================"

# Files to remove (legacy Predictathon scripts)
DELETE_FILES=(
    "src/brapi_utils.py"
    "src/fetch_accessions.py"
    "src/fetch_t3_genotypes.py"
    "src/clean_historical_phenotypes.py"
    "src/load_t3_csv.py"
    "src/make_grain_yield_pheno.py"
    "src/filter_genotypes.py"
    "src/parse_vcf.py"
    "src/impute_genotypes.py"
    "src/build_grm.py"
    "src/grm.py"
    "src/cv_protocols.py"
    "src/predictathon_main.py"
)

# Optional files (delete only if you no longer preprocess phenotypes)
OPTIONAL_FILES=(
    "src/preprocess_data.py"
    "geno_names.txt"
    "pheno_names.txt"
    "Trait Abbreviations - Wheat.csv"
)

# Directories to remove
DELETE_DIRS=(
    "src/__pycache__"
)

echo ""
echo "=== DRY RUN: The following files/directories WOULD be deleted ==="
for f in "${DELETE_FILES[@]}"; do
    [[ -f "$f" ]] && echo "FILE: $f"
done
for f in "${OPTIONAL_FILES[@]}"; do
    [[ -f "$f" ]] && echo "OPTIONAL: $f"
done
for d in "${DELETE_DIRS[@]}"; do
    [[ -d "$d" ]] && echo "DIR: $d"
done

echo ""
read -p "Proceed with deletion? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Aborted. No files were deleted."
    exit 0
fi

echo ""
echo "=== Deleting files ==="
for f in "${DELETE_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        rm "$f"
        echo "Deleted: $f"
    fi
done

echo ""
echo "=== Deleting optional files (if present) ==="
for f in "${OPTIONAL_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        rm "$f"
        echo "Deleted optional: $f"
    fi
done

echo ""
echo "=== Deleting directories ==="
for d in "${DELETE_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        rm -rf "$d"
        echo "Deleted directory: $d"
    fi
done

echo ""
echo "======================================"
echo " Cleanup complete!"
echo "======================================"