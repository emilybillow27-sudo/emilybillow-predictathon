###############################################
#   T3/Wheat Predictathon — Full Pipeline
#   Clean, cycle‑free, uses internal filtering
###############################################

rule all:
    input:
        "submission_output/cv1_results.csv",
        "submission_output/cv1_scatter.png",
        "submission_output/cv1_foldwise_accuracy.png"


############################################################
# 1. Merge VCFs → geno_merged_raw.csv
############################################################
rule merge_vcfs:
    output:
        "data/processed/geno_merged_raw.csv"
    shell:
        """
        python src/merge_vcfs.py
        """


############################################################
# 2. Build modeling matrix
############################################################
rule build_modeling_matrix:
    input:
        "data/processed/geno_merged_raw.csv"
    output:
        "data/processed/modeling_matrix.csv"
    shell:
        """
        python src/modeling_matrix.py
        """


############################################################
# 3. Merge environment data
############################################################
rule merge_env:
    input:
        "data/processed/modeling_matrix.csv"
    output:
        "data/processed/modeling_matrix_with_env.csv"
    shell:
        """
        python src/merge_env_into_modeling_matrix.py
        """


############################################################
# 4. Modeling (BLUPs → GRM → CV1 → predictions)
#    Uses your existing filtering logic inside main.py
############################################################
rule modeling:
    input:
        pheno="data/processed/modeling_matrix_with_env.csv",
        geno="data/processed/geno_merged_raw.csv"
    output:
        "submission_output/cv1_results.csv"
    shell:
        """
        python src/main.py
        """


############################################################
# 5. Visualization
############################################################
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