# app.py
# CA PSA Herbaceous (HER) API — uses RAP 16-day provisional (afgNPP/pfgNPP) vs PSA normals CSV
# Render env vars expected:
#   EE_SERVICE_ACCOUNT        (e.g., finefuel@finefuelloading.iam.gserviceaccount.com)
#   EE_PRIVATE_KEY_FILE       (/etc/secrets/ee-key.json)
#   PSA_NORMALS_CSV           (default: psa_HER_norm_CA_v3.csv)
#   EE_COLLECTION_16D_PROV    (default: projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional)
#   CACHE_TTL                 (seconds; default 900)

# app.py
# CA PSA Herbaceous (HER) API — uses RAP 16-day provisional (afgNPP/pfgNPP) vs PSA normals CSV
# Render env vars expected:
#   EE_SERVICE_ACCOUNT        (e.g., finefuel@finefuelloading.iam.gserviceaccount.com)
#   EE_PRIVATE_KEY_FILE       (/etc/secrets/ee-key.json)
#   PSA_NORMALS_CSV           (default: psa_HER_norm_CA_v3.csv)
#   EE_COLLECTION_16D_PROV    (default: projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional)
#   CACHE_TTL                 (seconds; default 900)

from __future__ import annotations

import os
import json
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import ee

# ----------------------------
# Environment / Config
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
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "").strip()  # e.g. /etc/secrets/ee-key.json

NORMALS_CSV = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")

HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="CA PSA Herbaceous API", version="3.3.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Helpers
# ----------------------------
def safe_num(v):
    """Convert EE/Pandas NaN/inf/None to JSON-safe numbers or None."""
    try:
        if v is None:
            return None
        if isinstance(v, float):
            if v != v:  # NaN
                return None
            if v == float("inf") or v == float("-inf"):
                return None
        return float(v)
    except Exception:
        return None


def parse_gaccs_param(gaccs_raw: Optional[str]) -> Optional[List[str]]:
    if not gaccs_raw:
        return None
    vals = [s.strip() for s in gaccs_raw.split(",") if s.strip()]
    return vals or None


# ----------------------------
# Load normals at startup
# ----------------------------
def load_normals(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normals CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalize column names we rely on
    need = ["PSANationalCode","PSANAME", "GACCUnitID", "afgNPP_norm", "pfgNPP_norm", "HER_norm"]
    for col in need:
        if col not in df.columns:
            raise ValueError(f"Normals CSV missing column: {col}")
    # Ensure PSA/GACC are uppercased for robust joins
    df["PSA_KEY"] = df["PSANationalCode"].astype(str).str.upper().str.strip() + df["PSANAME"].astype(str).str.upper().str.strip() + "|" + df["GACCUnitID"].astype(str).str.upper().str.strip()
    return df[["PSA_KEY"] + need]


# ----------------------------
# Earth Engine init (once)
# ----------------------------
EE_READY = False
try:
    if EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE:
        credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_FILE)
        ee.Initialize(credentials)
    else:
        # Fall back to default (won't work on Render unless you did user auth)
        ee.Initialize()
    EE_READY = True
except Exception as e:
    EE_READY = False
    print("EE init failed:", e)

# Load normals (once)
NORMALS_DF: Optional[pd.DataFrame] = None
try:
    NORMALS_DF = load_normals(NORMALS_CSV)
    print(f"Loaded normals: {len(NORMALS_DF)} rows from {NORMALS_CSV}")
except Exception as e:
    print("Normals load failed:", e)


# ----------------------------
# PSA FeatureCollection (geometry) from ArcGIS service
# ----------------------------
def get_psa_fc(gaccs: Optional[List[str]]) -> ee.FeatureCollection:
    # Pull all PSAs as GeoJSON from service (server-side via EE makes life easier if we host a copy,
    # but here we fetch minimal fields + geometry client-side, then upload as EE.FeatureCollection)
    # To keep runtime low, we query only California (the service is national but we filter by GACC).
    import requests

    params = {
        "where": "1=1",
        "outFields": "PSANationalCode,PSANAME,GACCUnitID",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": 4326,
    }
    r = requests.get(PSA_FS_URL, params=params, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    gj = r.json()
    feats = gj.get("features", [])

    # Filter by GACC if provided
    if gaccs:
        gset = set([g.upper() for g in gaccs])
        feats = [f for f in feats if str(f.get("properties", {}).get("GACCUnitID", "")).upper() in gset]

    # Convert to EE FeatureCollection
    ee_feats = []
    for f in feats:
        props = f.get("properties", {})
        geom = f.get("geometry", None)
        if not geom:
            continue
        key = (str(props.get("PSANationalCode", "")).upper().strip()
               + "|" + str(props.get("PSANAME", "")).upper().strip()  + "|" + str(props.get("GACCUnitID", "")).upper().strip())
        props["PSA_KEY"] = key
        ee_geom = ee.Geometry(geom)
        ee_feats.append(ee.Feature(ee_geom, props))
    return ee.FeatureCollection(ee_feats)


# ----------------------------
# Compute latest flags
# ----------------------------
def compute_latest_flags(gaccs_list: Optional[List[str]]):
    if not EE_READY:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized on server")
    if NORMALS_DF is None or NORMALS_DF.empty:
        raise HTTPException(status_code=500, detail="Normals table not loaded")

    # Latest composite in the 16-day collection
    coll = ee.ImageCollection(EE_COLLECTION_16D_PROV).sort("system:time_start", False)
    latest = coll.first()
    # Select bands and build HER = afgNPP + pfgNPP
    img = latest.select(["afgNPP", "pfgNPP"])
    her = img.select("afgNPP").add(img.select("pfgNPP")).rename("HER")
    stack = img.addBands(her)  # bands: afgNPP, pfgNPP, HER

    # Reduce by PSA polygons
    psa_fc = get_psa_fc(gaccs_list)

    # IMPORTANT: reduceRegions does NOT accept bestEffort/maxPixels -> use tileScale
    # Mean per band over each PSA polygon at Landsat scale 30 m
    stats_fc = stack.reduceRegions(
        collection=psa_fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=4
    )

    stats = stats_fc.getInfo().get("features", [])

    # Convert to DataFrame
    rows = []
    for f in stats:
        props = f.get("properties", {}) or {}
        psa_key = str(props.get("PSA_KEY", "")).upper().strip()
        rows.append({
            "PSA_KEY": psa_key,
            "PSANationalCode": props.get("PSANationalCode")
            "PSANAME": props.get("PSANAME"),
            "GACCUnitID": props.get("GACCUnitID"),
            "afgNPP_latest": safe_num(props.get("afgNPP")),
            "pfgNPP_latest": safe_num(props.get("pfgNPP")),
            "HER_latest": safe_num(props.get("HER")),
        })
    latest_df = pd.DataFrame(rows)

    # Merge with normals
    merged = pd.merge(latest_df, NORMALS_DF, on="PSA_KEY", how="left")

    # Compute above_normal flag (HER only, as requested)
    merged["above_normal"] = (
        (merged["HER_latest"].fillna(0) > merged["HER_norm"].fillna(0)).astype(int)
    )

    # Build response rows
    out = []
    for _, r in merged.iterrows():
        out.append({
            "PSANationalCode": r.get("PSANationalCode"),
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
        "rows": out
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "CA PSA Herbaceous API: /psa_flags?gaccs=USCAOSCC,USCAONCC&pretty=1"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ee_initialized": EE_READY,
        "normals_csv": NORMALS_CSV,
    }

@app.get("/psa_flags")
def psa_flags(
    gaccs: Optional[str] = Query(None, description="Comma-separated GACCUnitID values (e.g. USCAOSCC,USCAONCC)"),
    pretty: Optional[int] = Query(0, description="Pretty-print JSON if 1"),
    pretty1: Optional[int] = Query(0, description="Alias for pretty=1")
):
    try:
        gacc_list = parse_gaccs_param(gaccs)
        payload = compute_latest_flags(gacc_list)

        do_pretty = bool(pretty) or bool(pretty1)
        if do_pretty:
            return PlainTextResponse(json.dumps(payload, indent=2), media_type="application/json")

        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Run (local)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

