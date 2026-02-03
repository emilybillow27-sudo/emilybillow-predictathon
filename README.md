This repository contains a reproducible workflow for preparing phenotype and genotype data, integrating NASA POWER environmental covariates, estimating genotype BLUPs, fitting a multi‑environment GBLUP (reaction‑norm model), and generating Predictathon submission files.

The pipeline is implemented in Python with a small set of modular scripts.

🌱 Overview
The workflow performs the following steps:

Merge raw VCF files into a unified genotype matrix

Build a modeling‑ready phenotype matrix

Fetch and aggregate NASA POWER environmental covariates

Merge phenotype and environment data

Estimate genotype BLUPs using a mixed model

Construct a genomic relationship matrix (GRM)

Fit a multi‑environment GBLUP model

genotype main effects

environment fixed effects

genotype × environment interaction

Run CV1 cross‑validation (environment‑based)

Generate CV0/CV00 predictions for Predictathon challenge trials

Write submission‑ready output files

All outputs are written to the submission_output/ folder.

🧬 Requirements
Python 3.10+

pandas, numpy, scipy

seaborn, matplotlib

scikit‑learn

tqdm

You also need:

raw VCF files

phenotype metadata

accession lists for Predictathon trials

🚀 Running the Pipeline
From the repository root:

bash
python src/main.py
This runs:

BLUP estimation

GRM construction

multi‑environment GBLUP

CV1 cross‑validation

final predictions

submission file generation

📂 Input Files
Place the following in data/raw/:

metadata.csv

raw VCF files (*.vcf or *.vcf.gz)

accession lists for each Predictathon trial

preprocessed_final.csv (cleaned phenotype file)

Environmental covariates are fetched automatically and stored in:

Code
data/processed/env_all_high_overlap.csv
The merged phenotype + environment matrix is stored in:

Code
data/processed/modeling_matrix_with_env.csv
📤 Output Files
All outputs appear in:

Code
submission_output/
This includes:

cv1_results.csv

cv1_scatter.png

cv1_foldwise_accuracy.png

CV0 and CV00 prediction files for each challenge trial

🗂 Repository Structure
Code
src/
    main.py                         # End-to-end pipeline
    models.py                       # Multi-environment GBLUP implementation
    blups.py                        # Genotype BLUP estimation
    merge_env_into_modeling_matrix.py
    fetch_power_env.py              # NASA POWER integration
    merge_vcfs.py
    visualize_cv1.py
    submission.py
    vcf_utils.py
Snakefile                           # Optional Snakemake workflow
run_pipeline.sh
data/
    raw/
    processed/
submission_output/
🌾 Notes
The pipeline is designed to be fully reproducible and modular.

Environment covariates are integrated at the trial level.

The modeling system uses a reaction‑norm GBLUP with G and G×E kernels.

CV1
    submission.py
    t3_io.py
    vcf_utils.py
Snakefile
run_pipeline.sh
data/raw/
data/processed/
