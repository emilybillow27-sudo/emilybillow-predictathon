import pandas as pd
import requests
import time

# Input
env_list_path = "data/processed/historical_env_list.csv"

# Output
out_path = "data/processed/historical_env_metadata.csv"

# T3 Wheat BRAPI base URL
BASE = "https://wheat.triticeaetoolbox.org/brapi/v2"

# Load study IDs
env_list = pd.read_csv(env_list_path)
env_list["study_db_id"] = env_list["study_db_id"].astype("Int64")

study_ids = env_list["study_db_id"].dropna().unique()

records = []

def safe_get(url):
    """Wrapper to handle API rate limits and errors."""
    for _ in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1)
    return None

for sid in study_ids:
    print(f"Fetching study {sid}...")

    # --- Fetch study metadata ---
    study_url = f"{BASE}/studies/{sid}"
    study_json = safe_get(study_url)

    if not study_json or "result" not in study_json:
        print(f"  Study {sid} not found.")
        continue

    result = study_json["result"]

    studyName = result.get("studyName")
    locationDbId = result.get("locationDbId")
    plantingDate = result.get("plantingDate")
    harvestDate = result.get("harvestDate")

    # --- Fetch location metadata ---
    locationName = None
    latitude = None
    longitude = None

    if locationDbId:
        loc_url = f"{BASE}/locations/{locationDbId}"
        loc_json = safe_get(loc_url)

        if loc_json and "result" in loc_json:
            loc = loc_json["result"]
            locationName = loc.get("locationName")
            latitude = loc.get("latitude")
            longitude = loc.get("longitude")

    records.append({
        "study_db_id": sid,
        "studyName": studyName,
        "locationDbId": locationDbId,
        "locationName": locationName,
        "plantingDate": plantingDate,
        "harvestDate": harvestDate,
        "latitude": latitude,
        "longitude": longitude
    })

    # Be polite to the API
    time.sleep(0.2)

# Save results
df = pd.DataFrame(records)
df.to_csv(out_path, index=False)

print(f"\n✓ Wrote {out_path} with {len(df)} rows.")
