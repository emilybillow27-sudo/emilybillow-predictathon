import pandas as pd
from pathlib import Path

PRED_DIR = Path("data/predictathon")
ENV = Path("data/processed/env_covariates.csv")
OUTPUT = Path("data/processed/modeling_matrix.csv")

def main():
    pheno_files = list(PRED_DIR.glob("*/training_pheno_merged.csv"))
    if not pheno_files:
        raise FileNotFoundError("No training_pheno_merged.csv files found.")

    phenos = []
    for f in pheno_files:
        df = pd.read_csv(f, low_memory=False)

        # Normalize genotype column
        if "germplasm_name" in df.columns:
            df = df.rename(columns={"germplasm_name": "germplasmName"})
        else:
            raise KeyError(f"{f} does not contain germplasm_name")

        # Normalize phenotype column
        if "value" not in df.columns:
            raise KeyError(f"{f} does not contain value column")

        # Add environment name from folder
        df["studyName"] = f.parent.name

        phenos.append(df[["studyName", "germplasmName", "value"]])

    pheno = pd.concat(phenos, ignore_index=True)

    # FIX: force numeric and drop corrupted rows
    pheno["value"] = pd.to_numeric(pheno["value"], errors="coerce")
    pheno = pheno.dropna(subset=["value"])

    # Aggregate to genotype-level phenotypes
    pheno_agg = (
        pheno.groupby(["studyName", "germplasmName"])["value"]
        .mean()
        .reset_index()
    )

    # Load environmental covariates
    env = pd.read_csv(ENV)

    # Merge phenotypes with environment covariates
    merged = pheno_agg.merge(env, on="studyName", how="inner")

    # Load genotype metadata
    geno_files = list(PRED_DIR.glob("*/accessions.csv"))
    geno = pd.concat([pd.read_csv(f) for f in geno_files]).drop_duplicates()

    # Normalize genotype column in metadata
    if "germplasm_name" in geno.columns:
        geno = geno.rename(columns={"germplasm_name": "germplasmName"})

    merged = merged.merge(geno, on="germplasmName", how="inner")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    print(f"Wrote {len(merged)} rows to {OUTPUT}")

if __name__ == "__main__":
    main()