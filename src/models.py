import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


# ---------------------------------------------------------
# GRM builder
# ---------------------------------------------------------
def build_grm_from_geno(geno_numeric):
    """
    geno_numeric: numeric-only genotype matrix (rows = lines, cols = markers)
    """
    X = geno_numeric.to_numpy(dtype=float)

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

    return G


# ---------------------------------------------------------
# Environment kernel builder
# ---------------------------------------------------------
def build_env_kernel(env_df, env_id_col="studyName"):
    ENV_COLS = [
        "T2M", "T2M_MAX", "T2M_MIN",
        "PRECTOTCORR", "RH2M", "WS2M",
        "ALLSKY_SFC_SW_DWN"
    ]

    env_unique = env_df.drop_duplicates(subset=[env_id_col]).copy()
    env_ids = env_unique[env_id_col].tolist()

    X_env = env_unique[ENV_COLS].to_numpy(dtype=float)

    # Impute missing
    col_means = np.nanmean(X_env, axis=0)
    inds = np.where(np.isnan(X_env))
    if inds[0].size > 0:
        X_env[inds] = np.take(col_means, inds[1])

    D = euclidean_distances(X_env, X_env)
    sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0

    K_E = np.exp(-(D ** 2) / (2 * sigma ** 2))

    return K_E, env_ids, ENV_COLS


# ---------------------------------------------------------
# Multi-environment GBLUP
# ---------------------------------------------------------
def fit_model(train_pheno, geno_numeric, geno_lines, env, G, model_type="me_gblup"):
    """
    train_pheno: DataFrame with ['germplasmName','studyName','value']
    geno_numeric: numeric-only genotype matrix
    geno_lines: list of germplasm names in same order as geno_numeric rows
    env: environment covariates
    G: full GRM
    """

    df = train_pheno.copy()

    # phenotype vector
    y = df["value"].astype(float).to_numpy()
    y_mean = y.mean()
    y_centered = y - y_mean

    # environment kernel
    K_E, env_ids, ENV_COLS = build_env_kernel(env, env_id_col="studyName")

    # map lines to genotype indices
    train_lines = df["germplasmName"].tolist()
    line_idx = [geno_lines.index(l) for l in train_lines]

    # subset G
    G_sub = G[np.ix_(line_idx, line_idx)]

    # map environments
    env_idx_map = {e: i for i, e in enumerate(env_ids)}
    env_idx = [env_idx_map[e] for e in df["studyName"].tolist()]

    # build K_E for observations
    n_obs = len(df)
    K_E_obs = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        ei = env_idx[i]
        K_E_obs[i, :] = K_E[ei, env_idx]

    # kernels
    K_G_obs = G_sub
    K_GE_obs = K_G_obs * K_E_obs

    w_g = 1.0
    w_ge = 1.0
    lambda_ridge = 1.0

    K_combined = (
        w_g * K_G_obs +
        w_ge * K_GE_obs +
        lambda_ridge * np.eye(n_obs)
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
        "lambda_ridge": lambda_ridge,
    }


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_for_trial(model, focal_trial, test_accessions, geno_numeric, geno_lines, env, G, model_type="me_gblup"):

    u = model["u"]
    train_lines = model["train_lines"]
    train_envs = model["train_envs"]
    y_mean = model["y_mean"]
    K_E = model["K_E"]
    env_ids = model["env_ids"]
    w_g = model["w_g"]
    w_ge = model["w_ge"]

    # map training lines
    train_line_idx = [geno_lines.index(l) for l in train_lines]

    # map environments
    env_idx_map = {e: i for i, e in enumerate(env_ids)}
    train_env_idx = [env_idx_map[e] for e in train_envs]

    # map test accessions
    test_idx = []
    for acc in test_accessions:
        if acc in geno_lines:
            test_idx.append(geno_lines.index(acc))
        else:
            test_idx.append(None)

    # K_G_test_train
    G_test_train = []
    for idx in test_idx:
        if idx is None:
            G_test_train.append(np.zeros(len(train_line_idx)))
        else:
            G_test_train.append(G[idx, train_line_idx])
    G_test_train = np.array(G_test_train)

    # K_E_test_train
    if focal_trial not in env_idx_map:
        K_E_test_train = np.ones((len(test_accessions), len(train_env_idx)))
    else:
        focal_env_idx = env_idx_map[focal_trial]
        K_E_test_train = K_E[focal_env_idx, train_env_idx]
        K_E_test_train = np.tile(K_E_test_train, (len(test_accessions), 1))

    # combine kernels
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
def cross_validate_model(train_pheno, geno_numeric, geno_lines, env, G, model_type="me_gblup", n_folds=5):

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

        model = fit_model(
            train_pheno=train_df,
            geno_numeric=geno_numeric,
            geno_lines=geno_lines,
            env=env,
            G=G,
            model_type=model_type
        )

        for env_name in test_envs:
            test_accessions = test_df.loc[
                test_df["studyName"] == env_name, "germplasmName"
            ].tolist()

            preds = predict_for_trial(
                model=model,
                focal_trial=env_name,
                test_accessions=test_accessions,
                geno_numeric=geno_numeric,
                geno_lines=geno_lines,
                env=env,
                G=G,
                model_type=model_type,
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