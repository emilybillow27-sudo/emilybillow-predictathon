import pandas as pd

# ---------------------------------------------------------
# Input paths
# ---------------------------------------------------------
env_list_path = "data/processed/historical_env_list.csv"
meta_path = "data/raw/pheno_processed.csv"
site_lookup_path = "data/processed/site_lookup_historical.csv"

# Output
out_path = "data/processed/historical_env_metadata_completed.csv"

# ---------------------------------------------------------
# Load environment list
# ---------------------------------------------------------
env_list = pd.read_csv(env_list_path)
env_list["study_db_id"] = env_list["study_db_id"].astype("Int64")

# ---------------------------------------------------------
# Load historical metadata (T3 phenotype export)
# ---------------------------------------------------------
meta = pd.read_csv(
    meta_path,
    encoding="latin1",
    engine="python",
    dtype=str,
    on_bad_lines="skip"
)

# Extract one row per study
meta_unique = meta[[
    "studyDbId",
    "studyName",
    "plantingDate",
    "harvestDate",
    "locationName"
]].drop_duplicates(subset=["studyDbId"])

meta_unique["studyDbId"] = meta_unique["studyDbId"].astype("Int64")

# ---------------------------------------------------------
# Clean location names
# ---------------------------------------------------------
meta_unique["locationName"] = (
    meta_unique["locationName"]
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.strip()
)

# Convert "Stillwater, OK" → "STILLWATER"
meta_unique["site"] = (
    meta_unique["locationName"]
    .str.split(",", n=1).str[0]
    .str.upper()
    .str.strip()
)

# ---------------------------------------------------------
# Load site lookup table
# ---------------------------------------------------------
lookup = pd.read_csv(
    site_lookup_path,
    header=None,
    names=["station", "latitude", "longitude", "coord_type", "site"]
)

lookup["site"] = lookup["site"].astype(str).str.upper().str.strip()

# ---------------------------------------------------------
# Merge env_list → metadata
# ---------------------------------------------------------
merged = env_list.merge(
    meta_unique,
    left_on="study_db_id",
    right_on="studyDbId",
    how="left"
)

# ---------------------------------------------------------
# DROP GHOST STUDIES (no metadata anywhere)
# ---------------------------------------------------------
ghost_ids = {
    6537, 7451, 7757, 8200, 9104, 9107, 9238, 9273,
    10310, 10775, 10813, 10819, 10836, 10983, 10986,
    14113, 14114, 14118, 14133, 14142
}

before = len(merged)
merged = merged[~merged["study_db_id"].isin(ghost_ids)]
after = len(merged)

print(f"\n✓ Dropped {before - after} ghost environments with no metadata.")

# ---------------------------------------------------------
# Clean dates
# ---------------------------------------------------------
def fix_date(x):
    try:
        return pd.to_datetime(x, errors="coerce").date()
    except Exception:
        return None

merged["plantingDate"] = merged["plantingDate"].apply(fix_date)
merged["harvestDate"] = merged["harvestDate"].apply(fix_date)

# ---------------------------------------------------------
# Merge site → coordinates
# ---------------------------------------------------------
merged = merged.merge(
    lookup[["site", "latitude", "longitude"]],
    on="site",
    how="left"
)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
missing = merged[merged["latitude"].isna() | merged["longitude"].isna()]

if len(missing) > 0:
    print("\n⚠ WARNING: Some environments are missing coordinates!")
    print(missing[["study_db_id", "locationName", "site"]].head(20))
else:
    print("\n✓ All environments have latitude/longitude.")

# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------
merged.to_csv(out_path, index=False)
print(f"✓ Wrote {out_path} with {len(merged)} rows.")