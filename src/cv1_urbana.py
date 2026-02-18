import pandas as pd
import numpy as np
from cyvcf2 import VCF
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

# ---------------------------------------------------------
# 1. Load phenotype data
# ---------------------------------------------------------
pheno = pd.read_csv("data/processed/preprocessed_final.csv")

urbana_trial = "YT_Urb_25"   # correct trial name

# Extract Urbana phenotypes
pheno_urb = pheno[pheno["trial"] == urbana_trial].copy()

# Training phenotypes = all trials except Urbana
pheno_train = pheno[pheno["trial"] != urbana_trial].copy()

# ---------------------------------------------------------
# 2. Load genotype matrix from new VCF
# ---------------------------------------------------------
vcf = VCF("fhb_analysis_2026_production_final.vcf.gz")

samples = np.array(vcf.samples)
geno_list = []

for variant in vcf:
    g = variant.genotypes
    # Convert GT to numeric dosage (0,1,2)
    dosages = np.array([
        gt[0] + gt[1] if gt[0] >= 0 and gt[1] >= 0 else np.nan
        for gt in g
    ])
    geno_list.append(dosages)

G = np.array(geno_list).T  # samples x markers
geno_df = pd.DataFrame(G, index=samples)

# ---------------------------------------------------------
# 3. Align genotype with phenotype
# ---------------------------------------------------------
train_acc = pheno_train["germplasmName"].unique()
urb_acc = pheno_urb["germplasmName"].unique()

geno_train = geno_df.loc[geno_df.index.intersection(train_acc)]
geno_urb = geno_df.loc[geno_df.index.intersection(urb_acc)]

# Drop markers with missing data in training
geno_train = geno_train.dropna(axis=1)
geno_urb = geno_urb[geno_train.columns]

# ---------------------------------------------------------
# 4. Fit model on training data
# ---------------------------------------------------------
y_train = (
    pheno_train.groupby("germplasmName")["value"]
    .mean()
    .loc[geno_train.index]
)

model = Ridge(alpha=1.0)
model.fit(geno_train.values, y_train.values)

# ---------------------------------------------------------
# 5. Predict Urbana accessions
# ---------------------------------------------------------
pred_urb = model.predict(geno_urb.values)

# ---------------------------------------------------------
# 6. Compute accuracy
# ---------------------------------------------------------
y_urb = (
    pheno_urb.groupby("germplasmName")["value"]
    .mean()
    .loc[geno_urb.index]
)

accuracy = pearsonr(pred_urb, y_urb)[0]

print("\n==============================")
print("   CV1 Accuracy: Urbana Trial")
print("==============================")
print(f"Accuracy (r): {accuracy:.4f}")

# Save predictions
out = pd.DataFrame({
    "germplasmName": geno_urb.index,
    "observed": y_urb.values,
    "predicted": pred_urb
})
out.to_csv("submission_output/cv1_results.csv", index=False)

print("\n✓ CV1 complete. Results saved to:")
print("  submission_output/cv1_results.csv")