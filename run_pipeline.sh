#!/bin/bash
set -euo pipefail

echo "======================================"
echo "  T3/Wheat Predictathon Pipeline"
echo "======================================"

TRIALS=(
    AWY1_DVPWA_2024
    TCAP_2025_MANKS
    25_Big6_SVREC_SVREC
    OHRWW_2025_SPO
    CornellMaster_2025_McGowan
    24Crk_AY2-3
    2025_AYT_Aurora
    YT_Urb_25
    STP1_2025_MCG
)

for trial in "${TRIALS[@]}"; do
    echo ""
    echo "======================================"
    echo "Processing trial: $trial"
    echo "======================================"

    PROC_DIR="data/predictathon/$trial/processed"
    GRM_PATH="$PROC_DIR/GRM.npy"

    # ---------------------------------------------------------
    # Step 1 — Preprocess genotypes (only if needed)
    # ---------------------------------------------------------
    if [ ! -f "$GRM_PATH" ]; then
        echo "[pipeline] No GRM found — preprocessing genotypes for $trial"
        python -m src.genotypes.preprocess_genotypes "$trial"
    else
        echo "[pipeline] GRM already exists — skipping genotype preprocessing"
    fi

    # ---------------------------------------------------------
    # Step 2 — Train model
    # ---------------------------------------------------------
    echo "[pipeline] Training model for $trial"
    python -m src.model.train_model "$trial"

    # ---------------------------------------------------------
    # Step 3 — CV0
    # ---------------------------------------------------------
    echo "[pipeline] Running CV0 for $trial"
    python src/model/cv0_predict.py \
        --config config.yaml \
        --trial "$trial" \
        --out "results/cv0_predictions/${trial}.csv"

    # ---------------------------------------------------------
    # Step 4 — CV00
    # ---------------------------------------------------------
    echo "[pipeline] Running CV00 for $trial"
    python src/model/cv00_predict.py \
        --config config.yaml \
        --trial "$trial" \
        --out "results/cv00_predictions/${trial}.csv"

    # ---------------------------------------------------------
    # Step 5 — Expected accuracy
    # ---------------------------------------------------------
    echo "[pipeline] Computing expected accuracy for $trial"
    python src/model/expected_accuracy.py \
        --config config.yaml \
        --trial "$trial" \
        --out "results/expected_accuracy/${trial}.csv"

done

echo ""
echo "======================================"
echo "Pipeline complete."
echo "======================================"