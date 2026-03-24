# Predictathon 2025 Genomic Prediction Pipeline
### Author: Emily Billow
### Institution: Colorado State University
### Project: GBLUP Pipeline for Predictathon 2025

## Overview
This repository contains a complete, end‑to‑end pipeline for generating CV0 and CV00 predictions for all nine Predictathon trials. The workflow performs phenotype cleaning, genotype preprocessing, GRM construction, mixed‑model training, and Predictathon‑compliant output generation.

All genotype and phenotype inputs originate from T3/Wheat and must be downloaded locally before running the pipeline.

## Key Features
- Unified phenotype cleaning and accession harmonization

- Genotype merging, platform harmonization, and GRM construction

- Mixed‑model GBLUP training using the VanRaden relationship matrix

- Challenge‑compliant CV0 and CV00 masking

- Per‑trial prediction and expected‑accuracy estimation

- Automatic generation of Predictathon submission folders


## Repository Structure
```
data/
  raw/                     # Raw VCFs, raw phenotypes, metadata (local only)
  processed/               # Unified phenotypes, mapped IDs
  predictathon/            # Per-trial genotype + phenotype folders

src/
  genotypes/               # VCF preprocessing, GRM construction
  model/                   # Training, CV0, CV00, expected accuracy
  utils/                   # Shared helpers

trained_models/            # Saved GRMs and trained models
results/                   # CV0, CV00, and EA outputs
submission/                # Predictathon-compliant final submission

config.yaml                # Trial, trait, and model settings
run_pipeline.sh            # End-to-end pipeline runner
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

## Required Inputs (Must Exist Locally)
These files must be present for the pipeline to run.
Large genotype files are not stored in GitHub and must be downloaded manually.

1. Phenotype Inputs
- data/processed/unified_training_pheno_mapped.csv: Unified phenotype table with mapped accessions and a trial column
- data/predictathon//training_pheno_merged.csv: Per‑trial merged phenotype file (optional but recommended)
3. Genotype Inputs (LFS‑sized, not stored in GitHub)
- data/predictathon//genotypes/*.vcf.gz: Raw genotype VCFs downloaded from T3/Wheat
These files are typically hundreds of MB to several GB and must remain local.

4. Metadata / Mapping Files
- missing_pnw_studies.txt: Used for phenotype harmonization
- Any accession‑mapping tables: Required for phenotype/genotype alignment

## Running the Pipeline
```
bash run_pipeline.sh
```
This will:

- Preprocess genotypes (if GRM not already built)
- Train the GBLUP model
- Generate CV0 predictions
- Generate CV00 predictions
- Compute expected accuracy
- Write outputs to results/ and submission/

## Force a Clean Rebuild
```
bash run_pipeline.sh --clean
```
Removes cached GRMs, models, and predictions before rerunning.

## Outputs
All final Predictathon‑formatted outputs are written to:

```
submission/
  <TRIAL>/
    CV0/
    CV00/
```
Each folder contains:
- Predictions
- Accessions used
- Trials used for training

## Contact
Emily Billow
Graduate Research Assistant
Soil & Crop Sciences, Colorado State University

Emily.Billow@colostate.edu
