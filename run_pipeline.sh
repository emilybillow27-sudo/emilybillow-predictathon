#!/bin/bash
set -euo pipefail

echo "======================================"
echo "   T3/Wheat Predictathon Pipeline"
echo "======================================"

# --------------------------------------------------
# Cleaning option
# --------------------------------------------------
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning workspace..."

    rm -rf data/processed
    rm -rf trained_models
    rm -rf results
    rm -rf predictathon_submission

    echo "Clean-all complete."
    exit 0
fi

# --------------------------------------------------
# Optional: accuracy flag
# --------------------------------------------------
WITH_ACC=false
if [[ "${1:-}" == "--with-accuracy" ]]; then
    WITH_ACC=true
fi

# --------------------------------------------------
# Step 1: Environment pipeline (cached)
# --------------------------------------------------
echo ""
echo "======================================"
echo " Step 1: Environment pipeline (cached)"
echo "======================================"

HIST_WEATHER=data/processed/env_historical_standardized.csv

if [[ -f "$HIST_WEATHER" ]]; then
    echo "✓ Historical weather already exists — skipping weather fetch."
else
    echo "Building historical env list + metadata..."
    python src/extract_historical_env_list.py
    python src/build_historical_env_metadata.py

    echo "Fetching historical weather..."
    python src/fetch_historical_weather.py
fi

echo "Standardizing Predictathon envs..."
python src/standardize_env_covariates.py

echo "Merging environment covariates..."
python src/merge_env_covariates.py

# --------------------------------------------------
# Step 2: Run Snakemake pipeline
# --------------------------------------------------
echo ""
echo "======================================"
echo " Step 2: Running Snakemake pipeline"
echo "======================================"

snakemake -j 8 --rerun-incomplete --keep-going

# --------------------------------------------------
# Step 3: Optional expected accuracy
# --------------------------------------------------
if [[ "$WITH_ACC" == true ]]; then
    echo ""
    echo "======================================"
    echo " Step 3: Compute expected accuracy"
    echo "======================================"
    python src/compute_expected_accuracy.py
fi

# --------------------------------------------------
# Done
# --------------------------------------------------
echo ""
echo "======================================"
echo " Pipeline complete!"
echo "======================================"