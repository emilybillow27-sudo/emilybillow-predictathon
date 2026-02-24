#!/usr/bin/env python3

import os
import numpy as np
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "trained_models")
TRAINING_CACHE = os.path.join(MODEL_DIR, "training_metadata.pkl")

OUTPUT_GENO = os.path.join(MODEL_DIR, "geno_numeric_imputed.npy")
OUTPUT_LINES = os.path.join(MODEL_DIR, "geno_lines_imputed.pkl")


def main():
    print("\n=== Loading curated genotype metadata ===")

    with open(TRAINING_CACHE, "rb") as f:
        meta = pickle.load(f)

    geno_numeric = meta["geno_numeric"]
    geno_lines = list(meta["geno_lines_ordered"])

    print(f"Loaded genotype matrix: {geno_numeric.shape[0]} lines × {geno_numeric.shape[1]} markers")

    # ---------------------------------------------------------
    # Save genotype matrix exactly as curated
    # ---------------------------------------------------------
    np.save(OUTPUT_GENO, geno_numeric)

    with open(OUTPUT_LINES, "wb") as f:
        pickle.dump(geno_lines, f)

    print("\n✓ Genotype preprocessing complete.")
    print(f"Saved numeric matrix: {OUTPUT_GENO}")
    print(f"Saved line order:     {OUTPUT_LINES}")


if __name__ == "__main__":
    main()