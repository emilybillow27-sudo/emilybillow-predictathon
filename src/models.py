import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


# ---------------------------------------------------------
# GRM builder
# ---------------------------------------------------------
def build_grm_from_geno(geno_df):
    """
    Build a genomic relationship matrix G from a wide genotype DataFrame.

    Assumes:
        - geno_df.index = germplasmName
        - columns = marker1, marker2, ...
    """
    geno_lines = geno_df.index.tolist()
    X = geno_df.to_numpy(dtype=float)

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


# ---------------------------------------------------------
# Environment kernel builder
# ---------------------------------------------------------
def build_env_kernel(env_df, env_id_col="studyName"):
    """
    Build an environment similarity kernel K_E from env covariates.
    Handles missing values via column-mean imputation.
    """

    ENV_COLS = [
        "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOTCORR", "RH2M", "WS2M",
        "ALLSKY_SFC_SW_DWN"
    ]

    env_unique = env_df.drop_duplicates(subset=[env_id_col]).copy()
    env_ids = env_unique[env_id_col].tolist()

    # Extract matrix
    X_env = env_unique[ENV_COLS].to_numpy(dtype=float)

    # Impute missing values
    col_means = np.nanmean(X_env, axis=0)
    inds = np.where(np.isnan(X_env))
    if inds[0].size > 0:
        X_env[inds] = np.take(col_means, inds[1])

    # Compute distances
    D = euclidean_distances(X_env, X_env)

    # Scale parameter
    sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0

    # Gaussian kernel
    K_E = np.exp(-(D ** 2) / (2 * sigma ** 2))

    return K_E, env_ids, ENV_COLS


# ---------------------------------------------------------
# Multi-environment GBLUP with G, E, and GxE kernels
# ---------------------------------------------------------
def fit_model(train_pheno, geno, env, G, model_type="me_gblup"):
    """
    Multi-environment GBLUP with:
        - Genomic kernel K_G
        - Environment kernel K_E
        - GxE kernel K_GE = K_G ⊗ K_E (approximated via elementwise product)
    """

    # Merge phenotype with environment covariates
    df = train_pheno.merge(env, on=["studyName"], how="left")

    # Explicit phenotype column
    if "value" not in df.columns:
        raise ValueError("Phenotype column 'value' not found in phenotype DataFrame.")
    y = df["value"].astype(float).to_numpy()
    y_mean = y.mean()
    y_centered = y - y_mean

    # Environment kernel
    K_E, env_ids, ENV_COLS = build_env_kernel(env, env_id_col="studyName")

    # Map lines in pheno to genotype indices
    geno_lines = geno.index.tolist()
    train_lines = df["germplasmName"].tolist()
    line_idx = [geno_lines.index(l) for l in train_lines]

    # Subset G to training lines
    G_sub = G[np.ix_(line_idx, line_idx)]

    # Map environments in pheno to env kernel indices
    env_idx_map = {e: i for i, e in enumerate(env_ids)}
    env_idx = [env_idx_map[e] for e in df["studyName"].tolist()]

    # Build full K_E for observations
    n_obs = len(df)
    K_E_obs = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        ei = env_idx[i]
        K_E_obs[i, :] = K_E[ei, env_idx]

    # Genomic kernel for observations
    K_G_obs = G_sub

    # GxE kernel
    K_GE_obs = K_G_obs * K_E_obs

    # Kernel weights
    w_g = 1.0
    w_ge = 1.0
    lambda_ridge = 1.0

    K_combined = (
        w_g * K_G_obs
        + w_ge * K_GE_obs
        + lambda_ridge * np.eye(n_obs)
    )

    try:
        u = np.linalg.solve(K_combined, y_centered)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(K_combined, y_centered, rcond=None)[0]

    return {
        "train_lines": train_lines,
        "train_envs": df["studyName"].tolist(),
        "u": u,
        "geno_lines": geno_lines,
        "G_full": G,
        "y_mean": y_mean,
        "env_cols": ENV_COLS,
        "env_df": env,
        "K_E": K_E,
        "env_ids": env_ids,
        "w_g": w_g,
        "w_ge": w_ge,
        "lambda_ridge": lambda_ridge
    }


# ---------------------------------------------------------
# Prediction for a trial
# ---------------------------------------------------------
def predict_for_trial(model, focal_trial, test_accessions, geno, env, G, model_type="me_gblup"):

    u = model["u"]
    train_lines = model["train_lines"]
    train_envs = model["train_envs"]
    geno_lines = model["geno_lines"]
    y_mean = model["y_mean"]
    K_E = model["K_E"]
    env_ids = model["env_ids"]
    w_g = model["w_g"]
    w_ge = model["w_ge"]

    geno_lines_list = list(geno.index)

    # Map training lines/envs
    train_line_idx = [geno_lines_list.index(l) for l in train_lines]
    env_idx_map = {e: i for i, e in enumerate(env_ids)}
    train_env_idx = [env_idx_map[e] for e in train_envs]

    # Build K_G_test_train
    test_idx = []
    for acc in test_accessions:
        if acc in geno_lines_list:
            test_idx.append(geno_lines_list.index(acc))
        else:
            test_idx.append(None)

    G_test_train = []
    for idx in test_idx:
        if idx is None:
            G_test_train.append(np.zeros(len(train_line_idx)))
        else:
            G_test_train.append(G[idx, train_line_idx])
    G_test_train = np.array(G_test_train)

    # Environment index for focal trial
    if focal_trial not in env_idx_map:
        K_E_test_train = np.ones((len(test_accessions), len(train_env_idx)))
    else:
        focal_env_idx = env_idx_map[focal_trial]
        K_E_test_train = K_E[focal_env_idx, train_env_idx]
        K_E_test_train = np.tile(K_E_test_train, (len(test_accessions), 1))

    # Combine kernels
    K_G_test = G_test_train
    K_GE_test = G_test_train * K_E_test_train

    K_test_train = w_g * K_G_test + w_ge * K_GE_test

    g_hat = K_test_train @ u
    y_hat = y_mean + g_hat

    return pd.DataFrame({
        "germplasmName": test_accessions,
        "pred": y_hat
    })


# ---------------------------------------------------------
# CV1 cross-validation
# ---------------------------------------------------------
def cross_validate_model(train_pheno, geno, env, G, model_type="me_gblup", n_folds=5):

    pheno_col = "value"
    if pheno_col not in train_pheno.columns:
        raise ValueError("Phenotype column 'value' not found.")

    envs = train_pheno["studyName"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(envs)
    folds = np.array_split(envs, n_folds)

    results = []

    for fold_id, test_envs in enumerate(folds, start=1):
        test_envs = list(test_envs)

        train_df = train_pheno[~train_pheno["studyName"].isin(test_envs)]
        test_df = train_pheno[train_pheno["studyName"].isin(test_envs)]

        model = fit_model(train_df, geno, env, G)

        for env_name in test_envs:
            test_accessions = test_df.loc[
                test_df["studyName"] == env_name, "germplasmName"
            ].tolist()

            preds = predict_for_trial(
                model=model,
                focal_trial=env_name,
                test_accessions=test_accessions,
                geno=geno,
                env=env,
                G=G
            )

            merged = (
                test_df[test_df["studyName"] == env_name][["germplasmName", pheno_col]]
                .merge(preds, on="germplasmName", how="left")
                .rename(columns={pheno_col: "value"})
            )
            merged["fold"] = fold_id
            merged["studyName"] = env_name

            results.append(merged)

    return pd.concat(results, ignore_index=True)