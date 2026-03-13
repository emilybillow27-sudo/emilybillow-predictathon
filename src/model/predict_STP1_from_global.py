#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accessions",
        type=str,
        required=True,
        help="Text file listing STP1 accessions to predict."
    )
    parser.add_argument(
        "--grm",
        type=str,
        default="data/processed/global_union/GRM_global_union.npy"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="data/processed/global_union/G_global_union_samples.txt"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="trained_models/global_union_model/final_model.joblib"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load GRM + sample order
    # ---------------------------------------------------------
    print("[predict_STP1_from_global] Loading GRM...")
    G = np.load(args.grm)

    with open(args.samples) as f:
        grm_samples = [ln.strip() for ln in f]

    line_to_idx = {ln: i for i, ln in enumerate(grm_samples)}

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    print("[predict_STP1_from_global] Loading model...")
    model = joblib.load(args.model)

    # ---------------------------------------------------------
    # Load STP1 accessions
    # ---------------------------------------------------------
    accs = [ln.strip() for ln in open(args.accessions)]

    # ---------------------------------------------------------
    # Predict for all lines
    # ---------------------------------------------------------
    print("[predict_STP1_from_global] Predicting...")
    u_hat = model.predict_from_grm(G)

    preds = []
    for acc in accs:
        if acc in line_to_idx:
            preds.append(u_hat[line_to_idx[acc]])
        else:
            preds.append(0.0)  # fallback for non-genotyped lines

    df = pd.DataFrame({"germplasmName": accs, "pred": preds})
    df.to_csv(args.out, index=False)

    print(f"[predict_STP1_from_global] Saved predictions → {args.out}")


if __name__ == "__main__":
    main()