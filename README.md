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
