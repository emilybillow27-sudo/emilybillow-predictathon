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

    rm -f data/processed/historical_env_list.csv
    rm -f data/processed/historical_env_metadata.csv
    rm -f data/processed/env_historical.csv
    rm -f data/processed/env_historical_standardized.csv
    rm -f data/processed/env_covariates.csv
    rm -f data/processed/env_covariates_standardized.csv
    rm -f data/processed/env_all_standardized.csv

    rm -f trained_models/GRM.npy
    rm -f trained_models/geno_numeric_imputed.npy
    rm -f trained_models/geno_lines_imputed.pkl
    rm -f trained_models/final_model.joblib

    rm -rf predictathon_submission
    rm -f historical_trial_accuracy.csv
    rm -f predictathon_expected_accuracy.csv

    echo "Clean-all complete."
    exit 0
fi


# --------------------------------------------------
# Step 1–4: Environment pipeline (cached)
# --------------------------------------------------
echo ""
echo "======================================"
echo " Step 1–4: Environment pipeline (cached)"
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
# Step 5: Preprocess genotypes (cached)
# --------------------------------------------------
echo ""
echo "======================================"
echo " Step 5: Preprocess genotypes"
echo "======================================"

GENO_NUM=trained_models/geno_numeric_imputed.npy
GENO_LINES=trained_models/geno_lines_imputed.pkl

if [[ -f "$GENO_NUM" && -f "$GENO_LINES" ]]; then
    echo "✓ Genotypes already imputed — skipping."
else
    python src/preprocess_genotypes.py
fi


# --------------------------------------------------
# Step 6: Train ME-GBLUP model (cached)
# --------------------------------------------------
echo ""
echo "======================================"
echo " Step 6: Train ME-GBLUP model"
echo "======================================"

MODEL=trained_models/final_model.joblib

if [[ -f "$MODEL" ]]; then
    echo "✓ Model already trained — skipping."
else
    python src/train_model.py
fi


# --------------------------------------------------
# Step 7: Expected accuracy (optional)
# --------------------------------------------------
if [[ "${1:-}" == "--with-accuracy" ]]; then
    echo ""
    echo "======================================"
    echo " Step 7: Compute expected accuracy"
    echo "======================================"
    python src/compute_expected_accuracy.py
fi


# --------------------------------------------------
# Step 8: Build Predictathon submission (always rerun)
# --------------------------------------------------
echo ""
echo "======================================"
echo " Step 8: Build Predictathon submission"
echo "======================================"
python src/build_predictathon_submission.py


echo ""
echo "======================================"
echo " Pipeline complete!"
echo "======================================"