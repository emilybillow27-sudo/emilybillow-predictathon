import numpy as np
import pandas as pd


def compute_genotype_blups(pheno_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute genotype BLUPs from a long-format phenotype DataFrame
    using a simple random-intercept model:

        y = mu + u + e
        u ~ N(0, sigma_u I)
        e ~ N(0, sigma_e I)

    This is intentionally minimal and numerically stable.
    """

    df = pheno_long.copy()

    # Keep only what we need
    df = df[["value", "germplasmName"]].copy()
    df = df[df["value"].notna()].copy()

    # print("BLUP input shape after subsetting:", df.shape)
    # print("BLUP input columns:", df.columns.tolist())

    y = df["value"].values

    # Global mean (fixed intercept)
    mu = np.mean(y)

    # Random effects: genotype incidence
    geno_codes, geno_index = pd.factorize(df["germplasmName"])
    n_geno = len(geno_index)

    Z = np.zeros((len(df), n_geno), dtype=float)
    Z[np.arange(len(df)), geno_codes] = 1.0

    # Simple variance split
    sigma_y = np.var(y)
    if not np.isfinite(sigma_y) or sigma_y == 0:
        raise ValueError("Phenotype variance is zero or non-finite.")

    sigma_e = 0.5 * sigma_y
    sigma_u = 0.5 * sigma_y
    lam = sigma_e / sigma_u  # ridge term

    # Solve (Z'Z + λI) u = Z'(y - mu)
    ZtZ = Z.T @ Z + lam * np.eye(n_geno)

    try:
        ZtZ_inv = np.linalg.inv(ZtZ)
    except np.linalg.LinAlgError:
        ZtZ_inv = np.linalg.pinv(ZtZ)

    rhs_u = Z.T @ (y - mu)
    u = ZtZ_inv @ rhs_u

    blup_values = mu + u  # genotype-specific means shrunk toward mu

    return pd.DataFrame(
        {
            "germplasmName": geno_index,
            "blup": blup_values,
        }
    )