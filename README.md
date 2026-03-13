# T3/Wheat Prediction Challenge Pipeline

This repository contains a fully reproducible genomic prediction pipeline developed for the 2026 T3/Wheat Prediction Challenge. All genotype and phenotype data were downloaded directly from T3/Wheat using the website GUI.

## Features

- Phenotype cleaning and accession harmonization
- Genotype merging, platform harmonization, and imputation
- Environmental covariate integration
- Mixed‑model training with VanRaden GRM
- Challenge‑compliant CV0 and CV00 predictions
- Automatic generation of Predictathon submission folders

## Repository Structure

```
data/
  raw/            # VCFs, raw phenotypes, metadata
  processed/      # cleaned phenotypes, merged genotypes, modeling matrices
src/              # pipeline scripts (preprocessing, GRM, training, prediction)
trained_models/   # saved GRM, imputed genotypes, trained model
predictathon_submission/   # final submission folders
Snakefile         # workflow definition
run_pipeline.sh   # wrapper script
config.yaml       # trait, trial, and model settings
```

## Challenge Trials

AWY1_DVPWA_2024  
TCAP_2025_MANKS  
25_Big6_SVREC_SVREC  
OHRWW_2025_SPO  
CornellMaster_2025_McGowan  
24Crk_AY2-3  
2025_AYT_Aurora  
YT_Urb_25  
STP1_2025_MCG  

## Running the Pipeline
```
bash run_pipeline.sh
```

## Force a clean rebuild
```
bash run_pipeline --clean
```

Outputs are written to `submission_output/`
🌾 Predictathon 2025 Genomic Prediction Pipeline
Author: Emily Billow
Institution: Colorado State University
Project: Global GBLUP Pipeline for Predictathon 2025
Last Updated: March 2026

🌱 Overview
This repository contains a fully reproducible genomic prediction pipeline developed for the 2025 Wheat Predictathon. The workflow builds a global union genomic relationship matrix (GRM), trains a single global GBLUP model, and generates CV0 and CV00 predictions for all nine Predictathon trials.

The repository is structured for clarity, reproducibility, and long‑term maintainability. All intermediate artifacts, processed data, trained models, and final submission files are preserved.

🌾 Key Features
Unified phenotype processing across historical and Predictathon datasets

Global union GRM built from all available genotypes

Single global GBLUP model trained on all historical phenotypes

True CV0 and CV00 masking, following Predictathon definitions

Per‑trial predictions for all nine 2025 trials

Automated submission builder producing Predictathon‑compliant folders

Diagnostic reports for shrinkage, rank correlation, and zero‑prediction behavior

📁 Repository Structure
Code
emilybillow.github.io-8/
│
├── config.yaml                     # Central configuration for paths & parameters
├── Snakefile                       # Snakemake workflow (optional)
├── run_pipeline.sh                 # End-to-end pipeline runner
├── run_all_cv.py                   # Batch CV0/CV00 runner
│
├── data/
│   ├── raw/                        # Original Predictathon inputs (accessions, VCFs, metadata)
│   ├── processed/                  # Unified phenotypes, env covariates, global GRM
│   └── predictathon/               # Per-trial processed genotypes + merged phenotypes
│
├── src/
│   ├── env/                        # Environmental covariate processing
│   ├── genotypes/                  # Genotype preprocessing & VCF utilities
│   ├── model/                      # GBLUP training, prediction, diagnostics
│   ├── pipeline/                   # Predictathon-specific CV masking logic
│   ├── submission/                 # Submission builder & utilities
│   └── utils/                      # Shared helpers
│
├── results/
│   ├── cv0_predictions_yield/      # Final CV0 predictions (yield scale)
│   ├── cv00_predictions_yield/     # Final CV00 predictions (yield scale)
│   ├── expected_accuracy/          # Per-trial expected accuracy diagnostics
│   ├── genotype_coverage_diagnostics.csv
│   └── predictathon_genotype_coverage.csv
│
├── trained_models/
│   └── global_union_model/         # Final global GBLUP model
│
├── submission/                     # FINAL Predictathon submission (9 trials × CV0/CV00)
│   └── <TRIAL_NAME>/
│       ├── CV0/
│       │   ├── CV0_Predictions.csv
│       │   ├── CV0_Accessions.csv
│       │   └── CV0_Trials.csv
│       └── CV00/
│           ├── CV00_Predictions.csv
│           ├── CV00_Accessions.csv
│           └── CV00_Trials.csv
│
└── archive/                        # Safely stored intermediate & redundant outputs
🌾 Prediction Workflow
1. Genotype Processing
VCFs are filtered, encoded, and converted to numeric matrices

Per‑trial GRMs are computed

A global union GRM is built from all lines across trials

2. Phenotype Processing
Historical phenotypes are cleaned, standardized, and merged

Predictathon trial phenotypes are integrated where allowed

Unified phenotype file is produced

3. Model Training
A single global GBLUP model is trained using the global union GRM

Model is saved under trained_models/global_union_model/

4. Cross‑Validation (CV0 & CV00)
CV0: removes phenotypes from the focal trial only

CV00: removes phenotypes from the focal trial and all accessions in that trial

Predictions are generated for all nine trials

5. Submission Builder
Converts raw predictions into Predictathon‑compliant structure

Writes:

Predictions

Accessions list

Trials used for training

🌱 Interpreting Zero Predictions
Some trials (notably 24Crk_AY2‑3) contain accessions that:

appear only in that trial

have no phenotypes in any other environment

have no close genomic relatives with phenotypes

Under true CV0/CV00 masking, these lines have no usable training information, so their GBLUP breeding values correctly shrink to 0, and predicted yield shrinks to the global mean.

This is a data limitation, not a model failure.

📦 Final Submission
The final Predictathon submission is located in:

Code
submission/
Each trial contains:

CV0/ and CV00/

Predictions

Accessions list

Trials used for training

This structure is fully compliant with Predictathon requirements.

🧪 Reproducibility
To rerun the entire pipeline:

Code
bash run_pipeline.sh
Or run CV predictions only:

Code
python run_all_cv.py
To rebuild the submission folder:

Code
python src/submission/build_submission.py
🌻 Contact
For questions or collaboration:
Emily Billow  
Graduate Research Assistant
Soil & Crop Sciences, Colorado State University