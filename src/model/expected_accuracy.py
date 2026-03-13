#!/usr/bin/env python3

import argparse
import pathlib
import yaml
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so joblib can import src.model.models
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--trial", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load config + paths
    # ---------------------------------------------------------
    config = load_config(args.config)
    paths = config["paths"]

    trained_models_root = pathlib.Path(paths["trained_models_root"])
    unified_pheno_path = pathlib.Path(config["phenotypes"]["unified_training"])

    trial = args.trial
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load unified phenotype file
    # ---------------------------------------------------------
    if not unified_pheno_path.exists():
        raise SystemExit(f"Unified phenotype file not found: {unified_pheno_path}")

    pheno = pd.read_csv(unified_pheno_path, low_memory=False)

    # Identify trial column
    trial_cols = [
        c for c in pheno.columns
        if "trial" in c.lower() or "study" in c.lower()
    ]
    if not trial_cols:
        raise SystemExit("Could not infer trial column in unified phenotype file.")
    trial_col = trial_cols[0]

    # Subset to this trial
    ph = pheno[pheno[trial_col] == trial].copy()

    # If no phenotypes → expected accuracy not applicable
    if ph.empty:
        df = pd.DataFrame({
            "trial": [trial],
            "expected_accuracy": ["not_applicable"]
        })
        df.to_csv(out_path, index=False)
        print(f"[expected_accuracy] No phenotypes for {trial} → not_applicable")
        return

    # Standardize accession column
    if "germplasm_name" in ph.columns:
        ph = ph.rename(columns={"germplasm_name": "germplasmName"})
    if "germplasmName" not in ph.columns:
        raise SystemExit("Unified phenotype file must contain 'germplasmName'.")

    # Ensure numeric phenotype
    if "value" not in ph.columns:
        raise SystemExit("Unified phenotype file must contain a 'value' column.")

    ph["value"] = pd.to_numeric(ph["value"], errors="coerce")
    ph = ph.dropna(subset=["value"])

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model_path = trained_models_root / trial / "final_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    # ---------------------------------------------------------
    # Load GRM + line order
    # ---------------------------------------------------------
    grm_path = trained_models_root / trial / "GRM.npy"
    lines_path = trained_models_root / trial / "GRM_lines.txt"

    if not grm_path.exists():
        raise SystemExit(f"GRM not found: {grm_path}")
    if not lines_path.exists():
        raise SystemExit(f"GRM_lines.txt not found: {lines_path}")

    G = np.load(grm_path)
    with open(lines_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # ---------------------------------------------------------
    # Compute predictions for all lines
    # ---------------------------------------------------------
    if hasattr(model, "predict_from_grm"):
        u_hat = model.predict_from_grm(G)
    else:
        u_hat = model.predict(G)

    # ---------------------------------------------------------
    # Align phenotype with model line order
    # ---------------------------------------------------------
    ph_map = dict(zip(ph["germplasmName"], ph["value"]))
    y = np.array([ph_map.get(ln, np.nan) for ln in lines])

    mask = ~np.isnan(y)
    if mask.sum() == 0:
        df = pd.DataFrame({
            "trial": [trial],
            "expected_accuracy": ["not_applicable"]
        })
        df.to_csv(out_path, index=False)
        print(f"[expected_accuracy] No overlapping phenotype lines for {trial} → not_applicable")
        return

    # ---------------------------------------------------------
    # Compute expected accuracy = corr(u_hat, y)
    # ---------------------------------------------------------
    corr = np.corrcoef(u_hat[mask], y[mask])[0, 1]

    df = pd.DataFrame({
        "trial": [trial],
        "expected_accuracy": [float(corr)]
    })
    df.to_csv(out_path, index=False)

    print(f"[expected_accuracy] {trial}: expected_accuracy = {corr:.4f}")


if __name__ == "__main__":
    main()