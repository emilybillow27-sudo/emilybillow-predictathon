import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pheno_path = os.path.join(ROOT, "data", "processed", "modeling_matrix.csv")
env_cov_path = os.path.join(ROOT, "data", "processed", "env_covariates.csv")

pheno = pd.read_csv(pheno_path)
env_cov = pd.read_csv(env_cov_path)

print("Phenotype rows:", len(pheno))
print("Env cov rows:", len(env_cov))

pheno_env = pheno.merge(env_cov, on="studyName", how="left")

out_path = os.path.join(ROOT, "data", "processed", "modeling_matrix_with_env.csv")
pheno_env.to_csv(out_path, index=False)

print(f"✓ Saved modeling matrix with env covariates to {out_path}")
