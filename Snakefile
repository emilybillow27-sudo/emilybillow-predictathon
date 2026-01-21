###############################################
# Internal pipeline paths
###############################################

PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"

MODELING_MATRIX = f"{PROCESSED_DIR}/modeling_matrix.csv"
MERGED_GENO     = f"{PROCESSED_DIR}/geno_merged_raw.csv"
PHENO_GENO_MERGED = f"{PROCESSED_DIR}/train_pheno_overlap.csv"

###############################################
# Rule: merge_genotypes (VCF → geno_merged_raw.csv)
###############################################

rule merge_genotypes:
    output:
        MERGED_GENO
    shell:
        """
        python src/merge_vcfs.py
        """

###############################################
# Rule: modeling_matrix (phenotype + metadata → modeling_matrix.csv)
###############################################

rule modeling_matrix:
    input:
        pheno = f"{PROCESSED_DIR}/preprocessed_final.csv",
        metadata = f"{RAW_DIR}/metadata.csv"
    output:
        MODELING_MATRIX
    shell:
        """
        python src/modeling_matrix.py
        """

###############################################
# Rule: merge_pheno_geno (modeling_matrix + genotypes → train_pheno_overlap.csv)
###############################################

rule merge_pheno_geno:
    input:
        pheno = MODELING_MATRIX,
        geno  = MERGED_GENO
    output:
        PHENO_GENO_MERGED
    shell:
        """
        export PYTHONPATH="${{PYTHONPATH:-}}:src"

        python - << 'EOF'
from genotype_utils import load_genotype_matrix, merge_pheno_geno
import pandas as pd

pheno = pd.read_csv("{input.pheno}")
geno = load_genotype_matrix("{input.geno}")

merged = merge_pheno_geno(pheno, geno)
merged.to_csv("{output}", index=False)
EOF
        """

###############################################
# Rule: modeling (CV0 + CV00 → submission_output/)
###############################################

rule modeling:
    input:
        PHENO_GENO_MERGED
    output:
        directory("submission_output")
    shell:
        """
        python src/main.py
        """

###############################################
# Rule: visualize_cv1 (cv1_results.csv → cv1_scatter.png, cv1_foldwise_accuracy.png)
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