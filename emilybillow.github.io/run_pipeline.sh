#!/bin/bash
set -euo pipefail

echo "======================================"
echo "  T3/Wheat Predictathon Pipeline"
echo "======================================"

# ---------------------------------------------------------
# Optional cleaning step
# Usage: ./run_pipeline.sh --clean
# ---------------------------------------------------------
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning workspace..."

    rm -f data/processed/modeling_matrix.csv
    rm -f data/processed/geno_merged_raw.csv
    rm -f data/processed/train_pheno_overlap.csv
    rm -rf ../submission_output

    echo "Clean-all complete."
    exit 0
fi

# ---------------------------------------------------------
# Step 1: Modeling (CV1 + predictions)
# ---------------------------------------------------------
echo "Running modeling..."
snakemake modeling -p -j1
echo "--------------------------------------"

# ---------------------------------------------------------
# Step 2: CV1 visualization
# ---------------------------------------------------------
echo "Generating CV1 visualizations..."
snakemake visualize_cv1 -p -j1
echo "--------------------------------------"

# ---------------------------------------------------------
# Step 3: Full DAG (ensures everything is up to date)
# ---------------------------------------------------------
echo "Running full pipeline..."
snakemake -p -j1

echo "======================================"
echo " Pipeline complete!"
echo "======================================"