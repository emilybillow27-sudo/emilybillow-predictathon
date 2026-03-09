import pandas as pd
import glob
import os

# Where the merged phenotype files live
BASE = "data/predictathon"

study_ids = set()

# Loop through each trial folder
for trial in glob.glob(os.path.join(BASE, "*")):
    merged_path = os.path.join(trial, "training_pheno_merged.csv")
    if not os.path.isfile(merged_path):
        continue

    try:
        df = pd.read_csv(merged_path)
    except Exception as e:
        print(f"Skipping {merged_path} due to error: {e}")
        continue

    if "study_db_id" not in df.columns:
        print(f"No study_db_id column in {merged_path}")
        continue

    ids = df["study_db_id"].dropna().unique()
    study_ids.update(ids)

print(f"Total unique study_db_id: {len(study_ids)}")

# Save the list
out_df = pd.DataFrame({"study_db_id": sorted(study_ids)})
out_df.to_csv("data/processed/historical_env_list.csv", index=False)

print("✓ Wrote data/processed/historical_env_list.csv")