# app.py
# CA PSA Herbaceous (HER) API — uses RAP 16-day provisional (afgNPP/pfgNPP) vs PSA normals CSV
# Render env vars expected:
#   EE_SERVICE_ACCOUNT        (e.g., finefuel@finefuelloading.iam.gserviceaccount.com)
#   EE_PRIVATE_KEY_FILE       (/etc/secrets/ee-key.json)
#   PSA_NORMALS_CSV           (default: psa_HER_norm_CA_v3.csv)
#   EE_COLLECTION_16D_PROV    (default: projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional)
#   CACHE_TTL                 (seconds; default 900)

# app.py
# CA PSA Herbaceous (HER) API — RAP 16-day provisional (afgNPP/pfgNPP) vs PSA normals CSV
# Env vars expected on Render (or local):
#   EE_SERVICE_ACCOUNT      e.g. finefuel@finefuelloading.iam.gserviceaccount.com
#   EE_PRIVATE_KEY_FILE     e.g. /etc/secrets/ee-key.json
#   PSA_NORMALS_CSV         defaults to psa_HER_norm_CA_v3.csv
#   EE_COLLECTION_16D_PROV  defaults to projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional
#   HTTP_TIMEOUT_SEC        defaults to 60
#   CORS_ALLOW_ORIGINS      defaults to "*"

from __future__ import annotations

import os
import json
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import ee
import requests

# ----------------------------
# Config / constants
# ----------------------------
PSA_FS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

EE_COLLECTION_16D_PROV = os.getenv(
    "EE_COLLECTION_16D_PROV",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional",
)

EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT", "").strip()
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "").strip()
NORMALS_CSV = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="CA PSA Herbaceous API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Utilities
# ----------------------------
def safe_num(v):
    """Convert to JSON-safe float; return None for NaN/inf/None."""
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN
            return None
        if f in (float("inf"), float("-inf")):
            return None
        return f
    except Exception:
        return None


def parse_gaccs_param(gaccs_raw: Optional[str]) -> Optional[List[str]]:
    if not gaccs_raw:
        return None
    vals = [s.strip().upper() for s in gaccs_raw.split(",") if s.strip()]
    return vals or None


# ----------------------------
# Load normals (CSV) at startup
#   Required columns:
#   PSANationalCode, PSANAME, GACCUnitID, afgNPP_norm, pfgNPP_norm, HER_norm
# ----------------------------
def load_normals(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normals CSV not found: {path}")
    df = pd.read_csv(path)

    required = [
        "PSANationalCode", "PSANAME", "GACCUnitID",
        "afgNPP_norm", "pfgNPP_norm", "HER_norm"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Normals CSV missing column: {col}")

    df["PSA_KEY"] = df["PSANationalCode"].astype(str).str.upper().str.strip()
    # Keep label fields for output, too
    return df[["PSA_KEY", "PSANationalCode", "PSANAME", "GACCUnitID",
               "afgNPP_norm", "pfgNPP_norm", "HER_norm"]]


# Initialize EE once
EE_READY = False
try:
    if EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE:
        creds = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_FILE)
        ee.Initialize(creds)
    else:
        ee.Initialize()
    EE_READY = True
except Exception as e:
    print("EE init failed:", e)

# Load normals once
NORMALS_DF: Optional[pd.DataFrame] = None
try:
    NORMALS_DF = load_normals(NORMALS_CSV)
    print(f"Loaded normals: {len(NORMALS_DF)} rows from {NORMALS_CSV}")
except Exception as e:
    print("Normals load failed:", e)

# ----------------------------
# Fetch PSA polygons from ArcGIS (filtered by GACC if provided)
# We only request the fields we need and always include PSANationalCode.
# ----------------------------
def get_psa_fc(gaccs: Optional[List[str]]) -> ee.FeatureCollection:
    params = {
        "f": "geojson",
        "outSR": 4326,
        "returnGeometry": "true",
        "outFields": "PSANationalCode,PSANAME,GACCUnitID",
        "where": "1=1",
    }

    if gaccs:
        # Robust where clause: GACCUnitID IN ('USCAOSCC','USCAONCC')
        quoted = ",".join([f"'{g}'" for g in gaccs])
        params["where"] = f"GACCUnitID IN ({quoted})"

    r = requests.get(PSA_FS_URL, params=params, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    gj = r.json()
    features = gj.get("features", [])

    ee_feats = []
    for f in features:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry")
        if not geom:
            continue

        psa_code = str(props.get("PSANationalCode", "")).upper().strip()
        if not psa_code:
            # Skip any polygon without a national code
            continue

        props_out = {
            "PSA_KEY": psa_code,                       # canonical join key
            "PSANationalCode": psa_code,
            "PSANAME": props.get("PSANAME"),
            "GACCUnitID": props.get("GACCUnitID"),
        }
        ee_feats.append(ee.Feature(ee.Geometry(geom), props_out))

    return ee.FeatureCollection(ee_feats)


# ----------------------------
# Core: compute latest flags vs normals
# ----------------------------
def compute_latest_flags(gaccs_list: Optional[List[str]]):
    if not EE_READY:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized on server.")
    if NORMALS_DF is None or NORMALS_DF.empty:
        raise HTTPException(status_code=500, detail="Normals table not loaded.")

    # Latest RAP 16-day composite
    coll = ee.ImageCollection(EE_COLLECTION_16D_PROV).sort("system:time_start", False)
    latest = coll.first()
    latest_date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    # Bands + HER
    img = latest.select(["afgNPP", "pfgNPP"])
    her = img.select("afgNPP").add(img.select("pfgNPP")).rename("HER")
    stack = img.addBands(her)  # afgNPP, pfgNPP, HER

    # PSA polygons (filtered by GACC if provided)
    psa_fc = get_psa_fc(gaccs_list)

    # Mean per polygon @ 30m; tileScale for large regions
    stats_fc = stack.reduceRegions(
        collection=psa_fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=4
    )
    stats = stats_fc.getInfo().get("features", [])

    # To DF
    latest_rows = []
    for f in stats:
        p = f.get("properties", {}) or {}
        latest_rows.append({
            "PSA_KEY": p.get("PSA_KEY"),
            "PSANationalCode": p.get("PSANationalCode"),
            "PSANAME": p.get("PSANAME"),
            "GACCUnitID": p.get("GACCUnitID"),
            "afgNPP_latest": safe_num(p.get("afgNPP")),
            "pfgNPP_latest": safe_num(p.get("pfgNPP")),
            "HER_latest": safe_num(p.get("HER")),
        })
    latest_df = pd.DataFrame(latest_rows)

    if latest_df.empty:
        return {
            "count": 0,
            "gaccs": gaccs_list or [],
            "collection": EE_COLLECTION_16D_PROV,
            "latest_composite": latest_date,
            "rows": []
        }

    # Merge with normals on PSA_KEY (PSANationalCode)
    merged = pd.merge(latest_df, NORMALS_DF, on="PSA_KEY", how="left", suffixes=("", "_norms"))

    # above_normal flag (HER)
    merged["above_normal"] = (
        (merged["HER_latest"].fillna(-1e9) > merged["HER_norm"].fillna(1e9)).astype(int)
    )

    # Build payload rows
    out = []
    for _, r in merged.iterrows():
        out.append({
            "PSA": r.get("PSANationalCode"),         # the canonical code
            "PSANAME": r.get("PSANAME"),
            "GACCUnitID": r.get("GACCUnitID"),
            "afgNPP_latest": safe_num(r.get("afgNPP_latest")),
            "pfgNPP_latest": safe_num(r.get("pfgNPP_latest")),
            "HER_latest":    safe_num(r.get("HER_latest")),
            "afgNPP_norm":   safe_num(r.get("afgNPP_norm")),
            "pfgNPP_norm":   safe_num(r.get("pfgNPP_norm")),
            "HER_norm":      safe_num(r.get("HER_norm")),
            "above_normal":  int(r.get("above_normal", 0)),
        })

    return {
        "count": len(out),
        "gaccs": gaccs_list or [],
        "collection": EE_COLLECTION_16D_PROV,
        "latest_composite": latest_date,
        "rows": out
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "CA PSA Herbaceous API — try /psa_flags?gaccs=USCAOSCC,USCAONCC&pretty=1"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ee_initialized": EE_READY,
        "normals_csv": NORMALS_CSV,
    }

@app.get("/psa_flags")
def psa_flags(
    gaccs: Optional[str] = Query(None, description="Comma-separated GACCUnitID values (e.g., USCAOSCC,USCAONCC)"),
    pretty: Optional[int] = Query(0, description="Pretty-print JSON if 1"),
    pretty1: Optional[int] = Query(0, description="Alias for pretty=1"),
):
    try:
        gacc_list = parse_gaccs_param(gaccs)
        payload = compute_latest_flags(gacc_list)

        if bool(pretty) or bool(pretty1):
            return PlainTextResponse(json.dumps(payload, indent=2), media_type="application/json")
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Local run
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
