import os
import time
import json
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# -----------------------
# Optional Earth Engine
# -----------------------
EE_READY = False
try:
    import ee  # google earth engine
    EE_READY = True
except Exception:
    EE_READY = False

APP_NAME = "CA PSA Herbaceous API"
VERSION = "3.3.1"

# ---------- Config ----------
PSA_SERVICE = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

# Provisional 16-day NPP (2025→present)
COL_PROV = "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional"

# CSV with normals you generated in Jupyter
NORMALS_CSV = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")

# Simplification and scale (balance speed/accuracy)
SIMPLIFY_TOL_METERS = 120.0
REDUCE_SCALE_METERS = 90
TILESCALE = 4

# Cache TTL (seconds)
CACHE_TTL = int(os.getenv("CACHE_TTL", "900"))  # 15 minutes default

# ---------- App ----------
app = FastAPI(title=APP_NAME, version=VERSION)

# In-memory cache
_cache: Dict[str, Dict[str, Any]] = {}

def cache_get(key: str) -> Optional[Any]:
    item = _cache.get(key)
    if not item:
        return None
    if time.time() - item["t"] > CACHE_TTL:
        _cache.pop(key, None)
        return None
    return item["v"]

def cache_set(key: str, value: Any) -> None:
    _cache[key] = {"t": time.time(), "v": value}

# ---------- EE init ----------
def init_ee_if_possible() -> bool:
    global EE_READY
    if not EE_READY:
        return False
    try:
        # If already initialized, this no-ops
        ee.Initialize()
        return True
    except Exception:
        pass
    # Try service account
    sa = os.getenv("EE_SERVICE_ACCOUNT")
    key_file = os.getenv("EE_PRIVATE_KEY_FILE")
    if sa and key_file and os.path.exists(key_file):
        try:
            credentials = ee.ServiceAccountCredentials(sa, key_file)
            ee.Initialize(credentials)
            return True
        except Exception:
            return False
    return False

# ---------- PSA helpers ----------
def fetch_psas_geojson() -> Dict[str, Any]:
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": 4326,
        "resultRecordCount": 5000,
        "returnExceededLimitFeatures": "true",
    }
    r = requests.get(PSA_SERVICE, params=params, timeout=90)
    r.raise_for_status()
    gj = r.json()
    if "features" not in gj:
        raise HTTPException(status_code=502, detail="Unexpected PSA response (no features).")
    return gj

def build_psa_featurecollection(gacc_filter: List[str]) -> "ee.FeatureCollection":
    """Return EE FeatureCollection of CA PSAs filtered by GACCUnitID and without 'No PSA Assigned'."""
    gj = fetch_psas_geojson()
    feats = gj.get("features", [])
    keep = []
    for f in feats:
        props = f.get("properties", {}) or {}
        gacc = (props.get("GACCUnitID") or "").upper()
        name = props.get("PSANAME") or ""
        if gacc in gacc_filter and "No PSA Assigned" not in name:
            keep.append(
                ee.Feature(
                    ee.Geometry(f["geometry"]).simplify(SIMPLIFY_TOL_METERS),
                    {"PSANAME": name, "GACCUnitID": gacc},
                )
            )
    if not keep:
        raise HTTPException(status_code=404, detail="No PSAs matched the filter.")
    return ee.FeatureCollection(keep)

# ---------- Normals ----------
def load_normals_table() -> pd.DataFrame:
    if not os.path.exists(NORMALS_CSV):
        raise HTTPException(
            status_code=500,
            detail=f"Normals CSV not found: {NORMALS_CSV}. Upload the file next to app.py or set PSA_NORMALS_CSV.",
        )
    df = pd.read_csv(NORMALS_CSV)
    # expected columns: PSANAME, GACCUnitID, afgNPP_norm, pfgNPP_norm, HER_norm
    # normalize key fields
    if "PSANAME" not in df.columns or "GACCUnitID" not in df.columns:
        raise HTTPException(status_code=500, detail="Normals CSV missing PSANAME/GACCUnitID columns.")
    df["GACCUnitID"] = df["GACCUnitID"].str.upper()
    # If HER_norm not present, compute from afg/pfg
    if "HER_norm" not in df.columns:
        if "afgNPP_norm" in df.columns and "pfgNPP_norm" in df.columns:
            df["HER_norm"] = df["afgNPP_norm"].fillna(0) + df["pfgNPP_norm"].fillna(0)
        else:
            raise HTTPException(status_code=500, detail="Normals CSV lacks HER_norm and afg/pfg columns.")
    return df[["PSANAME", "GACCUnitID", "HER_norm"]].copy()

# ---------- Live latest vs normal ----------
def compute_latest_flags_for(gaccs: List[str]) -> Dict[str, Any]:
    """Return per-PSA latest HER, HER_norm and flag for the given GACCUnitIDs."""
    if not init_ee_if_possible():
        raise HTTPException(
            status_code=503,
            detail="Earth Engine is not initialized. Set EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE.",
        )

    normals = load_normals_table()

    # Build PSA FeatureCollection
    fc = build_psa_featurecollection(gaccs)

    # Latest provisional image
    ic = ee.ImageCollection(COL_PROV).sort("system:time_start", False)
    latest = ic.first()
    if latest.getInfo() is None:
        raise HTTPException(status_code=502, detail="No latest image in provisional collection.")
    interval_date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    # HER = afgNPP + pfgNPP
    latest_her = latest.select("afgNPP").add(latest.select("pfgNPP")).rename("HER_latest")

    # One server-side reduceRegions
    tbl = latest_her.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=REDUCE_SCALE_METERS,
        tileScale=TILESCALE,
    ).getInfo()["features"]

    latest_rows = []
    for f in tbl:
        p = f.get("properties", {})
        latest_rows.append(
            {
                "PSANAME": p.get("PSANAME"),
                "GACCUnitID": (p.get("GACCUnitID") or "").upper(),
                "HER_latest": p.get("HER_latest"),
            }
        )

    latest_df = pd.DataFrame(latest_rows)
    # join with normals
    merged = latest_df.merge(normals, on=["PSANAME", "GACCUnitID"], how="left")
    # compute flag
    merged["HER_above_normal"] = (merged["HER_latest"] >= merged["HER_norm"]).astype(int)
    merged["IntervalDate_latest"] = interval_date

    # sort for readability
    merged = merged.sort_values(["GACCUnitID", "PSANAME"]).reset_index(drop=True)

    return {
        "interval_date": interval_date,
        "count": int(len(merged)),
        "rows": merged.fillna(value={"HER_latest": None, "HER_norm": None}).to_dict(orient="records"),
    }

def pl_points_from_count(n: int) -> int:
    # 2pts for 0–1; 4pts for 2–4; 6pts for 5–7; 8pts for 8–10; 10pts for ≥11
    if n <= 1:
        return 2
    if n <= 4:
        return 4
    if n <= 7:
        return 6
    if n <= 10:
        return 8
    return 10

def summarize_points(rows: List[Dict[str, Any]], gacc: str) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    sub = df[df["GACCUnitID"].str.upper() == gacc]
    affected = int(sub["HER_above_normal"].sum())
    total = int(len(sub))
    interval_date = rows[0]["IntervalDate_latest"] if rows else None
    return {
        "GACCUnitID": gacc,
        "total_psas": total,
        "affected_psas": affected,
        "points": pl_points_from_count(affected),
        "as_of_interval_date": interval_date,
    }

# ---------- Routes ----------
@app.get("/")
def root():
    return {"name": APP_NAME, "version": VERSION}

@app.get("/health")
def health():
    ok = init_ee_if_possible()
    return {"status": "ok", "ee_initialized": ok, "normals_csv": NORMALS_CSV, "cache_ttl_sec": CACHE_TTL}

@app.get("/psa_flags")
def psa_flags(
    gaccs: str = Query(..., description="Comma-separated GACCUnitID, e.g. USCAOSCC,USCAONCC"),
):
    """
    Per-PSA latest vs. normal (binary flag). Example:
    /psa_flags?gaccs=USCAOSCC,USCAONCC
    """
    gacc_list = [g.strip().upper() for g in gaccs.split(",") if g.strip()]
    cache_key = f"flags:{','.join(sorted(gacc_list))}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    payload = compute_latest_flags_for(gacc_list)
    cache_set(cache_key, payload)
    return payload

@app.get("/southops_pl_points")
def southops_pl_points():
    """Points for Southern Ops (USCAOSCC)."""
    flags = psa_flags(gaccs="USCAOSCC")
    summary = summarize_points(flags["rows"], "USCAOSCC")
    return summary

@app.get("/northops_pl_points")
def northops_pl_points():
    """Points for Northern Ops (USCAONCC)."""
    flags = psa_flags(gaccs="USCAONCC")
    summary = summarize_points(flags["rows"], "USCAONCC")
    return summary

@app.get("/ca_points_summary")
def ca_points_summary():
    """Points for both USCAOSCC and USCAONCC in one call."""
    flags = psa_flags(gaccs="USCAOSCC,USCAONCC")
    south = summarize_points(flags["rows"], "USCAOSCC")
    north = summarize_points(flags["rows"], "USCAONCC")
    return {"southops": south, "northops": north, "interval_date": flags["interval_date"]}
