#!/bin/bash
set -euo pipefail

echo "======================================"
echo "  T3/Wheat Predictathon Pipeline"
echo "======================================"

# Cleaning option
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning workspace..."

    rm -f data/processed/modeling_matrix.csv
    rm -f data/processed/geno_merged_raw.csv
    rm -f data/processed/train_pheno_overlap.csv
    rm -rf submission_output

    echo "Clean-all complete."
    exit 0
fi

# Modeling
echo "Running modeling..."
snakemake modeling -p -j1
echo "--------------------------------------"

# Full pipeline
echo "Running full pipeline..."
snakemake -p -j1

echo "======================================"
echo " Pipeline complete!"
echo "======================================"