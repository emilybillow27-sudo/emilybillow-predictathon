###############################################
#   T3/Wheat Predictathon — Full Pipeline
###############################################

rule all:
    input:
        "predictathon_submission/ALL_DONE.txt"


############################################################
# 1. Extract historical env list
############################################################
rule extract_env_list:
    output:
        "data/processed/historical_env_list.csv"
    shell:
        "python src/extract_historical_env_list.py"


############################################################
# 2. Build historical env metadata
############################################################
rule build_env_metadata:
    input:
        "data/processed/historical_env_list.csv"
    output:
        "data/processed/historical_env_metadata.csv"
    shell:
        "python src/build_historical_env_metadata.py"


############################################################
# 3. Fetch historical weather
############################################################
rule fetch_weather:
    input:
        "data/processed/historical_env_metadata.csv"
    output:
        "data/processed/env_historical_standardized.csv"
    shell:
        "python src/fetch_historical_weather.py"


############################################################
# 4. Standardize Predictathon env covariates
############################################################
rule standardize_predictathon_env:
    output:
        "data/processed/env_covariates_standardized.csv"
    shell:
        "python src/standardize_env_covariates.py"


############################################################
# 5. Merge env covariates
############################################################
rule merge_env:
    input:
        hist="data/processed/env_historical_standardized.csv",
        pred="data/processed/env_covariates_standardized.csv"
    output:
        "data/processed/env_all_standardized.csv"
    shell:
        "python src/merge_env_covariates.py"


############################################################
# 6. Preprocess genotypes
############################################################
rule preprocess_genotypes:
    output:
        "trained_models/geno_numeric_imputed.npy",
        "trained_models/geno_lines_imputed.pkl"
    shell:
        "python src/preprocess_genotypes.py"


############################################################
# 7. Train ME-GBLUP model
############################################################
rule train_model:
    input:
        "trained_models/geno_numeric_imputed.npy",
        "trained_models/geno_lines_imputed.pkl",
        "data/processed/env_all_standardized.csv"
    output:
        "trained_models/final_model.joblib",
        "trained_models/GRM.npy"
    shell:
        "python src/train_model.py"


############################################################
# 8. Build Predictathon submission (CV0 + CV00)
############################################################
rule build_submission:
    input:
        "trained_models/final_model.joblib",
        "trained_models/GRM.npy",
        "data/processed/env_all_standardized.csv"
    output:
        "predictathon_submission/ALL_DONE.txt"
    shell:
        """
        python src/build_predictathon_submission.py
        echo 'done' > predictathon_submission/ALL_DONE.txt
        """