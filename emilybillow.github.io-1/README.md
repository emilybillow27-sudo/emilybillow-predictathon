# T3/Wheat Prediction Challenge Pipeline

This repository contains a fully reproducible genomic prediction pipeline developed for the 2026 T3/Wheat Prediction Challenge. All genotype and phenotype data were downloaded directly from T3/Wheat using the website GUI.

## Features

- Mixed-model BLUP estimation
- VanRaden genomic relationship matrix
- Environment-based CV1 cross-validation
- Challenge-compliant CV0 and CV00 predictions
- Automatic generation of required submission folder structure

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

bash run_pipeline.sh


Outputs are written to `submission_output/`.
