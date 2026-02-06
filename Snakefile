###############################################
# Paths
###############################################

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

RAW_PHENO              = f"{RAW_DIR}/pheno_processed.csv"
CLEAN_PHENO            = f"{PROCESSED_DIR}/preprocessed_final.csv"
MODELING_MATRIX        = f"{PROCESSED_DIR}/modeling_matrix.csv"
ENV_COVARIATES         = f"{PROCESSED_DIR}/env_covariates.csv"
MODELING_MATRIX_WITH_ENV = f"{PROCESSED_DIR}/modeling_matrix_with_env.csv"
MERGED_GENO            = f"{PROCESSED_DIR}/geno_merged_raw.csv"

###############################################
# 0. Preprocess raw T3 phenotype export
###############################################

rule preprocess_phenotypes:
    input:
        RAW_PHENO
    output:
        CLEAN_PHENO
    run:
        import pandas as pd
        from src.phenotype_utils import (
            clean_raw_t3_file,
            standardize_yield_traits,
            collapse_yield_traits
        )

        raw = pd.read_csv(input[0], encoding="latin1")

        clean = clean_raw_t3_file(raw)
        clean = standardize_yield_traits(clean)
        clean = collapse_yield_traits(clean, target="grain_yield")

        clean.to_csv(output[0], index=False)

###############################################
# 1. Build environmental covariates
###############################################

rule build_env_covariates:
    input:
        metadata = f"{RAW_DIR}/metadata.csv"
    output:
        ENV_COVARIATES
    shell:
        """
        python src/aggregate_env.py \
            --metadata {input.metadata} \
            --out {output}
        """

###############################################
# 2. Build modeling matrix (phenotype + metadata)
###############################################

rule modeling_matrix:
    input:
        pheno    = CLEAN_PHENO,
        metadata = f"{RAW_DIR}/metadata.csv"
    output:
        MODELING_MATRIX
    shell:
        """
        python src/modeling_matrix.py
        """

###############################################
# 3. Merge environmental covariates into modeling matrix
###############################################

rule merge_env:
    input:
        matrix = MODELING_MATRIX,
        env    = ENV_COVARIATES
    output:
        MODELING_MATRIX_WITH_ENV
    shell:
        """
        python src/merge_env_into_modeling_matrix.py \
            --matrix {input.matrix} \
            --env {input.env} \
            --out {output}
        """

###############################################
# 4. Merge genotypes
###############################################

rule merge_genotypes:
    output:
        MERGED_GENO
    shell:
        """
        python src/merge_vcfs.py
        """

###############################################
# 5. Run modeling (main pipeline)
###############################################

rule modeling:
    input:
        pheno = MODELING_MATRIX_WITH_ENV,
        geno  = MERGED_GENO
    output:
        directory("submission_output")
    shell:
        """
        python src/main.py
        """

###############################################
# 6. Visualize CV1
###############################################

rule visualize_cv1:
    input:
        "submission_output/cv1_results.csv"
    output:
        "submission_output/cv1_scatter.png",
        "submission_output/cv1_foldwise_accuracy.png"
    shell:
        """
        python src/visualize_cv1.py
        """

###############################################
# Final target
###############################################

rule all:
    input:
        "submission_output",
        "submission_output/cv1_scatter.png",
        "submission_output/cv1_foldwise_accuracy.png"