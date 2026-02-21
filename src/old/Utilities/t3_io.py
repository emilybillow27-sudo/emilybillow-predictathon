# src/t3_io.py

import requests
import pandas as pd

BASE_URL = "https://wheat.triticeaetoolbox.org/brapi/v2"

def _get(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

# Germplasm
def get_germplasm_by_name(names):
    """Fetch germplasm metadata for accession names."""
    out = []
    for name in names:
        resp = _get("germplasm-search", params={"germplasmName": name})
        out.extend(resp.get("result", {}).get("data", []))
    return pd.DataFrame(out)

# Studies / Trials
def get_study_metadata(studyDbId):
    """Fetch trial metadata."""
    resp = _get(f"studies/{studyDbId}")
    result = resp.get("result", {})
    meta = {
        "studyDbId": studyDbId,
        "studyName": result.get("studyName"),
        "location": result.get("locationName"),
        "year": result.get("studyYear"),
        "designType": result.get("designType"),
        "plantingDate": result.get("plantingDate"),
        "harvestDate": result.get("harvestDate"),
        "plotWidth": result.get("plotWidth"),
        "plotLength": result.get("plotLength"),
        "fieldSize": result.get("fieldSize"),
        "description": result.get("studyDescription"),
    }
    return pd.DataFrame([meta])

# Observations
def get_observations(studyDbId):
    """Fetch plot-level phenotypes."""
    resp = _get("observations-search", params={"studyDbId": studyDbId})
    data = resp.get("result", {}).get("data", [])
    return pd.DataFrame(data)

# Variables
def get_variables():
    """Fetch trait metadata."""
    resp = _get("variables")
    vars = resp.get("result", {}).get("data", [])
    rows = []
    for v in vars:
        rows.append({
            "observationVariableDbId": v.get("observationVariableDbId"),
            "name": v.get("name"),
            "abbreviation": v.get("abbreviation"),
            "traitClass": v.get("traitClass"),
            "description": v.get("description"),
        })
    return pd.DataFrame(rows)

# Lists
def get_list_items(listDbId):
    """Fetch items from a T3 list."""
    resp = _get(f"lists/{listDbId}")
    return resp.get("result", {}).get("data", [])