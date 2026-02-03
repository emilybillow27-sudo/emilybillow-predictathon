Predictathon Pipeline
This repository contains a small workflow for preparing phenotype and genotype data, building a modeling matrix, fitting a GBLUP model, and generating Predictathon submission files. Everything is organized into a Snakemake pipeline with a few supporting Python scripts.

Overview
The pipeline does the following:
1. Merge raw VCF files into a single genotype matrix
2. Build a modeling‑ready phenotype matrix
3. Merge phenotype and genotype data
4. Fit the model and generate CV0/CV00 predictions
5. Produce CV1 diagnostic plots

*All outputs are written to the submission_output/ folder.

Requirements:
- Python 3.10+
- Snakemake
- Standard scientific Python packages (pandas, numpy, scipy, seaborn, matplotlib)
- tqdm, scikit‑learn
- Access to raw VCFs, metadata, and phenotype files

--------------------------------------------------------------------------------------------------------------------------------------------

Running the Pipeline

Full run:

bash
./run_pipeline.sh

This runs all Snakemake rules in order.

Clean workspace:

bash
./run_pipeline.sh --clean

This removes processed files and the submission folder.

Input Files:

Place the following in data/raw/:
- metadata.csv
- Raw VCF files (*.vcf or *.vcf.gz)
- Accession lists for each Predictathon trial
- preprocessed_final.csv (cleaned phenotype file)

Output Files:

The main outputs appear in:

submission_output/

This includes:
- cv1_results.csv
- cv1_scatter.png
- cv1_foldwise_accuracy.png
- CV0 and CV00 prediction files for each trial

Repository Structure
Code
src/
    main.py
    modeling_matrix.py
    merge_vcfs.py
    visualize_cv1.py
    models.py
    submission.py
    t3_io.py
    vcf_utils.py
Snakefile
run_pipeline.sh
data/raw/
data/processed/
data/
    raw/
    processed/
submission_output/

CV0 and CV00 prediction files for each Predictathon trial
