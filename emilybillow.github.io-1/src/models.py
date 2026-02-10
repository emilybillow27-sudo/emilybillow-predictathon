import numpy as np
import pandas as pd


# GRM builder
def build_grm_from_geno(geno_df):
    """
    Build a genomic relationship matrix G from a wide genotype DataFrame.
    """

    geno_lines = geno_df["germplasmName"].tolist()
    X = geno_df.drop(columns=["germplasmName"]).to_numpy(dtype=float)

    # Impute missing with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])

    # Remove monomorphic markers
    std = X.std(axis=0)
    keep = std > 0
    X = X[:, keep]

    if X.shape[1] == 0:
        raise ValueError("All markers are monomorphic after filtering.")

    # Center markers
    X_centered = X - X.mean(axis=0, keepdims=True)
    m = X_centered.shape[1]

    # VanRaden-like GRM
    G = (X_centered @ X_centered.T) / m

    return G, geno_lines


# Multi-environment GBLUP model
def fit_model(train_pheno, geno, env, G, model_type="me_gblup"):
    """
    Multi-environment GBLUP (reaction-norm model):
        y = Xb + Zu + Wv + e
    where:
        Xb  = fixed effects (environment covariates)
        Zu  = random genotype main effects (G kernel)
        Wv  = random genotype × environment interaction (G ⊗ E kernel)
    """

    # ---------------------------------------------------------
    # 1. Merge phenotype with environment covariates
    # ---------------------------------------------------------
    df = train_pheno.merge(env, on=["germplasmName", "studyName"], how="left")

    # Identify phenotype column
    pheno_cols = [
        c for c in df.columns
        if c not in ["germplasmName", "studyName", "traitName"]
        and c not in [
            "T2M", "T2M_MAX", "T2M_MIN",
            "PRECTOTCORR", "RH2M", "WS2M",
            "ALLSKY_SFC_SW_DWN"
        ]
    ]
    if len(pheno_cols) != 1:
        raise ValueError(f"Could not identify phenotype column. Found: {pheno_cols}")
    pheno_col = pheno_cols[0]

    # ---------------------------------------------------------
    # 2. Fixed effects: NASA POWER covariates
    # ---------------------------------------------------------
    ENV_COLS = [
        "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOTCORR", "RH2M", "WS2M",
        "ALLSKY_SFC_SW_DWN"
    ]

    X = df[ENV_COLS].to_numpy()

    # ---------------------------------------------------------
    # 3. Random effects: genotype main effects (G kernel)
    # ---------------------------------------------------------
    geno_lines = geno["germplasmName"].tolist()
    train_lines = df["germplasmName"].tolist()

    # Map phenotype rows to genotype indices
    idx = [geno_lines.index(l) for l in train_lines]
    G_sub = G[np.ix_(idx, idx)]

    # ---------------------------------------------------------
    # 4. Random effects: genotype × environment interaction
    # ---------------------------------------------------------
    env_ids = df["studyName"].astype("category").cat.codes.to_numpy()
    n_env = df["studyName"].nunique()

    # Build environment incidence matrix
    W_env = np.zeros((len(df), n_env))
    W_env[np.arange(len(df)), env_ids] = 1

    # GxE kernel = elementwise product of G_sub and environment similarity
    GE_kernel = G_sub * (W_env @ W_env.T)

    # ---------------------------------------------------------
    # 5. Solve mixed model using ridge approximation
    # ---------------------------------------------------------
    y = df[pheno_col].to_numpy()
    y_mean = y.mean()
    y_centered = y - y_mean

    # Combine kernels
    lambda_g = 1.0
    lambda_ge = 1.0

    K = G_sub + GE_kernel + lambda_g * np.eye(len(G_sub)) + lambda_ge * np.eye(len(G_sub))

    # Solve for u (genotype main effects)
    try:
        u = np.linalg.solve(K, y_centered)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(K, y_centered, rcond=None)[0]

    return {
        "train_lines": train_lines,
        "u": u,
        "geno_lines": geno_lines,
        "G_full": G,
        "y_mean": y_mean,
        "env_cols": ENV_COLS,
        "env_df": env,
    }


# Prediction
def predict_for_trial(model, focal_trial, test_accessions, geno, env, G, model_type="me_gblup"):
    """
    Predict using multi-environment GBLUP:
        y_hat ≈ y_mean + G_test,train * u
    (env effects are absorbed into y_mean in this approximation).
    """

    # Unpack model
    u = model["u"]
    train_lines = model["train_lines"]
    geno_lines = model["geno_lines"]
    y_mean = model["y_mean"]

    geno_lines_list = geno["germplasmName"].tolist()

    # Map test accessions to genotype indices
    test_idx = []
    for acc in test_accessions:
        if acc in geno_lines_list:
            test_idx.append(geno_lines_list.index(acc))
        else:
            test_idx.append(None)

    # Relationship between test and training lines
    train_idx = [geno_lines_list.index(l) for l in train_lines]

    G_test_train = []
    for idx in test_idx:
        if idx is None:
            G_test_train.append(np.zeros(len(train_lines)))
        else:
            G_test_train.append(G[idx, train_idx])

    G_test_train = np.array(G_test_train)

    g_hat = G_test_train @ u
    y_hat = y_mean + g_hat

    return pd.DataFrame({
        "germplasmName": test_accessions,
        "pred": y_hat
    })


# CV1 cross-validation (by environment)
def cross_validate_model(train_pheno, geno, env, G, model_type="me_gblup", n_folds=5):
    """
    Perform CV1 cross-validation:
    - Hold out entire environments (studyName) in each fold.
    """

    # Identify phenotype column
    pheno_cols = [
        c for c in train_pheno.columns
        if c not in ["germplasmName", "studyName", "traitName"]
    ]
    if len(pheno_cols) != 1:
        raise ValueError(f"Could not identify phenotype column. Found: {pheno_cols}")
    pheno_col = pheno_cols[0]

    studies = train_pheno["studyName"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(studies)

    folds = np.array_split(studies, n_folds)

    results = []

    for fold_id, test_studies in enumerate(folds, start=1):
        train_df = train_pheno[~train_pheno["studyName"].isin(test_studies)]
        test_df = train_pheno[train_pheno["studyName"].isin(test_studies)]

        model = fit_model(train_df, geno, env, G, model_type)

        for study in test_studies:
            test_accessions = test_df[test_df["studyName"] == study]["germplasmName"].tolist()

            preds = predict_for_trial(
                model=model,
                focal_trial=study,
                test_accessions=test_accessions,
                geno=geno,
                env=env,
                G=G,
                model_type=model_type,
            )

            merged = test_df[test_df["studyName"] == study][["germplasmName", pheno_col]].merge(
                preds, on="germplasmName", how="left"
            )
            merged = merged.rename(columns={pheno_col: "value"})
            merged["fold"] = fold_id
            merged["studyName"] = study

            results.append(merged)

    return pd.concat(results, ignore_index=True)