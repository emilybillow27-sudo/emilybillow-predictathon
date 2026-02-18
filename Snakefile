###############################################
#   T3/Wheat Predictathon — Full Pipeline
#   Clean, cycle‑free, train + predict split
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
# 4. Train model → trained_models/
#    Produces: final_model.joblib, GRM.npy, training_* files
############################################################
rule train_model:
    input:
        pheno="data/processed/modeling_matrix_with_env.csv",
        geno="data/processed/geno_merged_raw.csv"
    output:
        "trained_models/final_model.joblib",
        "trained_models/GRM.npy",
        "trained_models/training_pheno_used.csv",
        "trained_models/training_geno_used.csv",
        "trained_models/training_env_used.csv"
    shell:
        """
        python src/train_model.py
        """


############################################################
# 5. Predict using trained model → cv1_results.csv
############################################################
rule predict_model:
    input:
        model="trained_models/final_model.joblib",
        pheno="trained_models/training_pheno_used.csv",
        geno="trained_models/training_geno_used.csv",
        env="trained_models/training_env_used.csv",
        grm="trained_models/GRM.npy"
    output:
        "submission_output/cv1_results.csv"
    shell:
        """
        python src/predict_model.py
        """


############################################################
# 6. Visualization
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