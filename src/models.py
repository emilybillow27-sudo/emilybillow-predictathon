import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ------------------------------------------------------------
# 0. Build a stable VanRaden-like GRM from genotype matrix
# ------------------------------------------------------------

def build_grm_from_geno(geno_df):
    """
    Build a genomic relationship matrix G from a wide genotype DataFrame.

    Assumes:
      - geno_df has a 'germplasmName' column
      - all other columns are numeric marker genotypes (0/1/2 or dosages)
    """

    # Keep line order consistent
    geno_lines = geno_df["germplasmName"].tolist()

    # Extract marker matrix
    X = geno_df.drop(columns=["germplasmName"]).to_numpy(dtype=float)

    # Simple missing-data handling: impute column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])

    # Remove monomorphic markers (zero variance)
    std = X.std(axis=0)
    keep = std > 0
    X = X[:, keep]

    if X.shape[1] == 0:
        raise ValueError("All markers are monomorphic after filtering.")

    # Center markers by column mean
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Number of markers
    m = X_centered.shape[1]

    # VanRaden-like GRM: G = X_centered X_centered' / m
    G = (X_centered @ X_centered.T) / m

    return G, geno_lines


# ------------------------------------------------------------
# 1. Fit GBLUP model using stabilized mixed model equation
# ------------------------------------------------------------

def fit_model(train_pheno, geno, env, G, model_type="me_gblup"):
    """
    Fit a GBLUP model using the GRM and phenotype vector.
    Uses:
        u = (G + λI)^(-1) y
    """

    # Identify phenotype column
    pheno_cols = [
        c for c in train_pheno.columns
        if c not in ["germplasmName", "studyName", "traitName"]
    ]
    if len(pheno_cols) != 1:
        raise ValueError(f"Could not identify phenotype column. Found: {pheno_cols}")
    pheno_col = pheno_cols[0]

    # Genotyped lines
    geno_lines = geno["germplasmName"].tolist()

    # Training lines that have genotypes
    train_lines = [l for l in train_pheno["germplasmName"].unique() if l in geno_lines]

    # Build phenotype vector aligned to GRM
    y_raw = np.array([
        train_pheno.loc[train_pheno["germplasmName"] == l, pheno_col].mean()
        for l in train_lines
    ])

    # Store phenotype mean for rescaling predictions
    y_mean = y_raw.mean()

    # Center phenotype
    y = y_raw - y_mean

    # Subset GRM to training lines
    idx = [geno_lines.index(l) for l in train_lines]
    G_sub = G[np.ix_(idx, idx)]

    # Ridge penalty (λ) — increased for stability
    lambda_ = 1.0
    A = G_sub + lambda_ * np.eye(len(G_sub))

    # Solve for breeding values (safe solve)
    try:
        u = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(A, y, rcond=None)[0]

    return {
        "train_lines": train_lines,
        "u": u,
        "geno_lines": geno_lines,
        "G_full": G,
        "y_mean": y_mean,
    }


# ------------------------------------------------------------
# 2. Predict for a trial
# ------------------------------------------------------------

def predict_for_trial(model, focal_trial, test_accessions, geno, env, G, model_type="me_gblup"):
    """
    Predict breeding values for a list of accessions using:
        pred_i = g_i,train @ u + y_mean
    """

    train_lines = model["train_lines"]
    u = model["u"]
    geno_lines = model["geno_lines"]
    G_full = model["G_full"]
    y_mean = model["y_mean"]

    preds = []
    for acc in test_accessions:
        if acc not in geno_lines:
            preds.append(np.nan)
            continue

        i = geno_lines.index(acc)
        g_vec = G_full[i, [geno_lines.index(l) for l in train_lines]]

        pred = g_vec @ u + y_mean
        preds.append(pred)

    return pd.DataFrame({
        "germplasmName": test_accessions,
        "pred": preds
    })


# ------------------------------------------------------------
# 3. CV1 cross-validation with diagnostics
# ------------------------------------------------------------

def cross_validate_model(train_pheno, geno, env, G, model_type="me_gblup", n_folds=5):
    """
    Perform CV1 (leave-lines-out) cross-validation.
    Returns:
        germplasmName | value | pred | fold
    """

    # Identify phenotype column
    pheno_cols = [
        c for c in train_pheno.columns
        if c not in ["germplasmName", "studyName", "traitName"]
    ]
    if len(pheno_cols) != 1:
        raise ValueError(f"Could not identify phenotype column. Found: {pheno_cols}")
    pheno_col = pheno_cols[0]

    lines = train_pheno["germplasmName"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(lines), start=1):
        train_lines = lines[train_idx]
        test_lines = lines[test_idx]

        pheno_train = train_pheno[train_pheno["germplasmName"].isin(train_lines)]
        pheno_test = train_pheno[train_pheno["germplasmName"].isin(test_lines)]

        model = fit_model(pheno_train, geno, env, G, model_type)

        preds = predict_for_trial(
            model=model,
            focal_trial="CV1",
            test_accessions=test_lines,
            geno=geno,
            env=env,
            G=G,
            model_type=model_type,
        )

        # Diagnostics: variance comparison
        print(f"[Fold {fold}] Pred variance:", preds["pred"].var())
        print(f"[Fold {fold}] Value variance:", pheno_test[pheno_col].var())

        merged = pheno_test[["germplasmName", pheno_col]].merge(
            preds, on="germplasmName", how="inner"
        )
        merged = merged.rename(columns={pheno_col: "value"})
        merged["fold"] = fold

        results.append(merged)

    return pd.concat(results, ignore_index=True)