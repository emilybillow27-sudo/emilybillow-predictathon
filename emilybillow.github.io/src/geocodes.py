#!/usr/bin/env python3
import os
import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modeling_matrix_path = os.path.join(ROOT, "data", "processed", "modeling_matrix.csv")
lookup_path = os.path.join(ROOT, "data", "processed", "site_lookup.csv")
out_path = os.path.join(ROOT, "data", "processed", "historical_locations.csv")

# ---------------------------------------------------------
# Load study names
# ---------------------------------------------------------
ph = pd.read_csv(modeling_matrix_path)
studies = sorted(ph["studyName"].unique())
df = pd.DataFrame({"studyName": studies})

# ---------------------------------------------------------
# Extract site code from studyName
# ---------------------------------------------------------
def extract_code(studyName):
    parts = studyName.replace("_", "-").split("-")
    return parts[-1].upper()

df["CODE"] = df["studyName"].apply(extract_code)

# ---------------------------------------------------------
# Load curated lookup table
# ---------------------------------------------------------
lookup = pd.read_csv(lookup_path)
lookup["CODE"] = lookup["CODE"].str.upper()

# ---------------------------------------------------------
# Merge lookup into df
# ---------------------------------------------------------
df = df.merge(lookup, on="CODE", how="left")

matched = df["LATITUDE"].notna().sum()
print(f"Matched {matched} / {len(df)} trials using curated lookup table")

# ---------------------------------------------------------
# Identify unknown codes (not to be geocoded)
# ---------------------------------------------------------
unknown = df[df["LATITUDE"].isna()].copy()

if len(unknown) > 0:
    print("\nThe following site codes were NOT found in the lookup table:")
    print(sorted(unknown["CODE"].unique()))
    print("\nThese will NOT be geocoded and will remain missing.")

# ---------------------------------------------------------
# Save final table
# ---------------------------------------------------------
final = df[["studyName", "CODE", "LOCATION_NAME", "LATITUDE", "LONGITUDE"]]
final.to_csv(out_path, index=False)

print("\n✓ Saved historical trial locations to:")
print(out_path)