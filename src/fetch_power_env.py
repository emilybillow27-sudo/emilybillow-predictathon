#!/usr/bin/env python3
import os
import pandas as pd
import requests
from datetime import datetime
import re

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

loc_path   = os.path.join(ROOT, "data", "processed", "historical_locations.csv")
dates_path = os.path.join(ROOT, "data", "raw", "metadata.csv")
pheno_path = os.path.join(ROOT, "data", "processed", "modeling_matrix.csv")
acc_dir    = os.path.join(ROOT, "data", "raw", "accession_lists")

out_dir = os.path.join(ROOT, "data", "processed", "env")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Build Predictathon accession set
# ---------------------------------------------------------
acc_files = [f for f in os.listdir(acc_dir) if f.endswith(".txt")]

predictathon_accessions = set()
for f in acc_files:
    path = os.path.join(acc_dir, f)
    accs = pd.read_csv(path, header=None)[0].astype(str).str.upper()
    predictathon_accessions.update(accs)

print(f"Found {len(acc_files)} Predictathon accession-list trials")
print(f"Total unique Predictathon accessions: {len(predictathon_accessions)}")

# ---------------------------------------------------------
# 2. Compute overlap between historical trials and Predictathon accessions
# ---------------------------------------------------------
ph = pd.read_csv(pheno_path)
ph["germplasmName"] = ph["germplasmName"].astype(str).str.upper()

trial_groups = ph.groupby("studyName")["germplasmName"].apply(set)

overlap_counts = {
    trial: len(accs & predictathon_accessions)
    for trial, accs in trial_groups.items()
}

overlap_df = (
    pd.DataFrame.from_dict(overlap_counts, orient="index", columns=["n_overlapping_accessions"])
      .sort_values("n_overlapping_accessions", ascending=False)
)

# Keep trials with >=50 overlapping accessions
high_overlap_trials = overlap_df[overlap_df["n_overlapping_accessions"] >= 50].index.tolist()

print(f"\nTrials with >=50 overlapping accessions: {len(high_overlap_trials)}")

# ---------------------------------------------------------
# 3. Load metadata and filter to high-overlap trials
# ---------------------------------------------------------
loc = pd.read_csv(loc_path)
dates = pd.read_csv(dates_path)
dates = dates[["studyName", "plantingDate", "harvestDate"]]

meta = loc.merge(dates, on="studyName", how="left")
meta = meta[meta["studyName"].isin(high_overlap_trials)].copy()

print(f"\nTrials remaining after filtering by overlap: {len(meta)}")
if meta.empty:
    print("No trials satisfy the overlap threshold — exiting.")
    raise SystemExit

# ---------------------------------------------------------
# 4. Site coordinate dictionary (only the 10 needed sites)
# ---------------------------------------------------------
SITE_COORDS = {
    "COLKS": (39.395, -101.052),   # Colby, KS
    "HAYKS": (38.879, -99.322),    # Hays, KS
    "HUTKS": (38.060, -97.929),    # Hutchinson, KS
    "MANKS": (39.183, -96.571),    # Manhattan, KS

    "FTC":   (40.585, -105.084),   # Fort Collins, CO
    "GRY":   (40.423, -104.709),   # Greeley, CO

    "MASMI": (42.580, -84.443),    # Mason, MI
    "URB":   (40.110, -88.227),    # Urbana, IL
    "NEO":   (35.535, -97.481),    # Oklahoma City region, OK

    "SNYDER": (42.447, -76.475),   # Snyder Farm, Ithaca, NY
}

SITE_TO_STATE = {
    "COLKS": "KS",
    "HAYKS": "KS",
    "HUTKS": "KS",
    "MANKS": "KS",

    "FTC": "CO",
    "GRY": "CO",

    "MASMI": "MI",
    "URB": "IL",
    "NEO": "OK",

    "SNYDER": "NY",
}

# ---------------------------------------------------------
# 5. Helper: extract year robustly
# ---------------------------------------------------------
def extract_year(study):
    m = re.search(r"\d{2,4}", study)
    if not m:
        return None
    y = int(m.group())
    if y < 100:  # convert 22 → 2022
        y += 2000
    return y

# ---------------------------------------------------------
# 6. Coordinate inference
# ---------------------------------------------------------
def infer_coordinates(row):
    if pd.notna(row["LATITUDE"]) and pd.notna(row["LONGITUDE"]):
        return row

    study = row["studyName"]
    tokens = study.replace("_", "-").split("-")

    for t in tokens:
        t_up = ''.join([c for c in t.upper() if c.isalpha()])
        if t_up in SITE_COORDS:
            row["LATITUDE"], row["LONGITUDE"] = SITE_COORDS[t_up]
            return row

    return row

meta = meta.apply(infer_coordinates, axis=1)

# ---------------------------------------------------------
# 7. Planting/harvest date inference
# ---------------------------------------------------------
WINTER_STATES = {"CO", "KS", "NE", "SD", "OK", "TX", "IL", "MI", "NY"}
SPRING_STATES = {"ND", "MT", "MN"}

def infer_dates(row):
    # If plantingDate already exists, keep it
    if pd.notna(row["plantingDate"]):
        return row

    study = row["studyName"]
    year = extract_year(study)
    if year is None:
        return row

    # Infer state from site code
    state = None
    tokens = study.replace("_", "-").split("-")
    for t in tokens:
        t_up = ''.join([c for c in t.upper() if c.isalpha()])
        if t_up in SITE_TO_STATE:
            state = SITE_TO_STATE[t_up]
            break

    if state is None:
        state = "KS"  # safe default for winter wheat

    # Winter wheat
    if state in WINTER_STATES:
        row["plantingDate"] = datetime(year, 9, 25)
        row["harvestDate"]  = datetime(year + 1, 7, 1)
        return row

    # Spring wheat
    if state in SPRING_STATES:
        row["plantingDate"] = datetime(year, 4, 25)
        row["harvestDate"]  = datetime(year, 8, 15)
        return row

    # Fallback
    row["plantingDate"] = datetime(year, 9, 25)
    row["harvestDate"]  = datetime(year + 1, 7, 1)
    return row

meta = meta.apply(infer_dates, axis=1)

# Convert to datetime
meta["plantingDate"] = pd.to_datetime(meta["plantingDate"], errors="coerce")
meta["harvestDate"]  = pd.to_datetime(meta["harvestDate"], errors="coerce")

# ---------------------------------------------------------
# 8. NASA POWER fetcher
# ---------------------------------------------------------
def fetch_power_daily(lat, lon, start_date, end_date):
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "community": "AG",
        "format": "JSON",
        "parameters": ",".join([
            "T2M", "T2M_MAX", "T2M_MIN",
            "PRECTOTCORR", "RH2M", "WS2M",
            "ALLSKY_SFC_SW_DWN"
        ])
    }

    r = requests.get(base, params=params)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df.index)
    df = df.reset_index(drop=True)
    return df

# ---------------------------------------------------------
# 9. Fetch weather for each trial
# ---------------------------------------------------------
all_env = []

for _, row in meta.iterrows():
    study = row["studyName"]
    lat   = row["LATITUDE"]
    lon   = row["LONGITUDE"]
    start = row["plantingDate"]
    end   = row["harvestDate"]

    if pd.isna(lat) or pd.isna(lon):
        print(f"Skipping {study}: missing coordinates")
        continue

    if pd.isna(start):
        print(f"Skipping {study}: missing planting date")
        continue

    if pd.isna(end):
        end = pd.Timestamp.today()

    print(f"\n=== Fetching NASA POWER for {study} ===")
    print(f"Location: {lat}, {lon}")
    print(f"Dates: {start.date()} → {end.date()}")

    try:
        df_env = fetch_power_daily(lat, lon, start, end)
    except Exception as e:
        print(f"  ERROR fetching {study}: {e}")
        continue

    df_env["studyName"] = study
    df_env.to_csv(os.path.join(out_dir, f"{study}_env.csv"), index=False)
    all_env.append(df_env)

# ---------------------------------------------------------
# 10. Save combined file
# ---------------------------------------------------------
if all_env:
    combined = pd.concat(all_env, ignore_index=True)
    combined.to_csv(os.path.join(ROOT, "data", "processed", "env_all_high_overlap.csv"), index=False)
    print("\n✓ Saved combined environmental covariates.")
else:
    print("\nNo environment data fetched — check metadata.")