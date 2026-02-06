import pandas as pd
import numpy as np

def compute_genotype_blups(pheno_long):
    df = pheno_long.copy()

    # Keep only needed columns
    keep = ["value", "studyName", "germplasmName"]
    df = df[keep].dropna()

    print("BLUP input shape after subsetting:", df.shape)
    print("BLUP input columns:", df.columns.tolist())

    y = df["value"].values

    # Fixed effects: intercept + study dummies
    study_dummies = pd.get_dummies(df["studyName"], drop_first=True)
    X = np.column_stack([np.ones(len(df)), study_dummies.values])

    # Random effects: genotype one-hot
    geno_codes, geno_index = pd.factorize(df["germplasmName"])
    n_geno = len(geno_index)

    Z = np.zeros((len(df), n_geno))
    Z[np.arange(len(df)), geno_codes] = 1

    # Variance components (simple REML-like estimates)
    # These don't need to be perfect for BLUP ranking
    sigma_e = np.var(y) * 0.5
    sigma_u = np.var(y) * 0.5

    # Henderson’s mixed model equations:
    # [X'X     X'Z] [b] = [X'y]
    # [Z'X  Z'Z+λI] [u]   [Z'y]
    lam = sigma_e / sigma_u

    XtX = X.T @ X
    XtZ = X.T @ Z
    ZtX = XtZ.T
    ZtZ = Z.T @ Z + lam * np.eye(n_geno)

    # Build and solve the block system
    top = np.hstack([XtX, XtZ])
    bottom = np.hstack([ZtX, ZtZ])
    LHS = np.vstack([top, bottom])

    RHS = np.concatenate([X.T @ y, Z.T @ y])

    sol = np.linalg.solve(LHS, RHS)

    # Extract solutions
    b = sol[:X.shape[1]]      # fixed effects
    u = sol[X.shape[1]:]      # random effects (BLUPs)

    # Put BLUPs on original scale (add intercept)
    intercept = b[0]
    blup_values = u + intercept

    blups = pd.DataFrame({
        "germplasmName": geno_index,
        "blup": blup_values
    })

    return blups