import pandas as pd

# Input files
env_list_path = "data/processed/historical_env_list.csv"
meta_path = "data/raw/metadata.csv"
site_lookup_path = "data/raw/site_lookup.csv"

# Output
out_path = "data/processed/historical_env_metadata.csv"

# Load study_db_id list
env_list = pd.read_csv(env_list_path)

# Convert floats like 9071.0 → ints like 9071
env_list["study_db_id"] = env_list["study_db_id"].astype("Int64")

# Load metadata
meta = pd.read_csv(meta_path)

# Load site lookup (lat/lon)
lookup = pd.read_csv(site_lookup_path)

# --- STEP 1: Join study_db_id → metadata ---
merged = env_list.merge(
    meta,
    left_on="study_db_id",
    right_on="studyDbId",
    how="left"
)

# --- STEP 2: Clean planting/harvest dates ---
def fix_date(x):
    try:
        return pd.to_datetime(x, errors="coerce").date()
    except:
        return None

merged["plantingDate"] = merged["plantingDate"].apply(fix_date)
merged["harvestDate"] = merged["harvestDate"].apply(fix_date)

# --- STEP 3: Join locationName → lat/lon ---
lookup_cols = lookup.columns

if "locationName" in lookup_cols:
    merged = merged.merge(lookup, on="locationName", how="left")
elif "town" in lookup_cols:
    merged = merged.merge(lookup, left_on="locationName", right_on="town", how="left")
else:
    raise ValueError("site_lookup.csv must contain either 'locationName' or 'town'.")

# --- STEP 4: Save ---
merged.to_csv(out_path, index=False)

print(f"✓ Wrote {out_path} with {len(merged)} rows.")