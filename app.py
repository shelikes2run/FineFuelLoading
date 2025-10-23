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
import math
import logging
from typing import Dict, List, Optional

import ee
import pandas as pd
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ------------- Settings & logging -------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ca-psa-herbaceous-api")

CORS_ALLOW = os.getenv("CORS_ALLOW_ORIGINS", "*")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))
EE_COLLECTION_16D_PROV = os.getenv(
    "EE_COLLECTION_16D_PROV",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional",
)
EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT", "")
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "")
NORMS_CSV_PATH = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")

PSA_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

app = FastAPI(title="CA PSA Herbaceous API", version="3.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOW] if CORS_ALLOW != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Earth Engine init -------------

EE_READY = False

def init_ee() -> None:
    global EE_READY
    if EE_READY:
        return
    if not EE_SERVICE_ACCOUNT or not EE_PRIVATE_KEY_FILE:
        raise RuntimeError(
            "EE credentials not configured. Set EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE."
        )
    credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_FILE)
    ee.Initialize(credentials)
    EE_READY = True
    log.info("Earth Engine initialized.")

# ------------- Normals CSV (afgNPP_norm/pfgNPP_norm/HER_norm) -------------

def load_norms_df() -> pd.DataFrame:
    if not os.path.exists(NORMS_CSV_PATH):
        raise RuntimeError(f"Normals CSV not found: {NORMS_CSV_PATH}")
    df = pd.read_csv(NORMS_CSV_PATH)
    # Normalize expected column names
    rename_map = {
        "PSANAME": "PSANAME",
        "GACCUnitID": "GACCUnitID",
        "afgNPP_norm": "afgNPP_norm",
        "pfgNPP_norm": "pfgNPP_norm",
        "HER_norm": "HER_norm",
    }
    df = df.rename(columns=rename_map)
    # keep only CA PSAs present in CSV
    cols = ["PSANAME", "GACCUnitID", "afgNPP_norm", "pfgNPP_norm", "HER_norm"]
    df = df[cols]
    return df

NORMS_DF = load_norms_df()

# ------------- Helpers -------------

def fetch_ca_psas_geojson() -> Dict:
    """Pull CA PSA polygons as GeoJSON (properties include PSANAME, GACCUnitID)."""
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        r = client.get(PSA_URL, params=params)
        r.raise_for_status()
        return r.json()

def filter_geojson_by_gacc(geo: Dict, gacc_list: Optional[List[str]]) -> Dict:
    if not gacc_list:
        return geo
    gset = {g.upper() for g in gacc_list}
    feats = []
    for f in geo.get("features", []):
        props = f.get("properties", {}) or {}
        gacc = (props.get("GACCUnitID") or "").upper()
        if gacc in gset:
            feats.append(f)
    return {"type": "FeatureCollection", "features": feats}

def latest_rap_image() -> ee.Image:
    """Get the latest 16-day RAP NPP composite and add HER band."""
    coll = ee.ImageCollection(EE_COLLECTION_16D_PROV).sort("system:time_start", False)
    img = coll.first()
    # band names in RAP v3: "afgNPP" (annual forb/grass) and "pfgNPP" (perennial forb/grass)
    afg = img.select("afgNPP")
    pfg = img.select("pfgNPP")
    her = afg.add(pfg).rename("HER")
    return afg.rename("afgNPP").addBands([pfg.rename("pfgNPP"), her])

def ee_mean_by_polygons(img: ee.Image, geojson_fc: Dict) -> List[Dict]:
    """
    Reduce mean per polygon. We keep band names as-is so properties
    in the result are 'afgNPP', 'pfgNPP', 'HER' (NOT *_mean).
    """
    fc = ee.FeatureCollection(geojson_fc)
    reduced = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30,            # 30 m
        maxPixels=1_000_000_000,
        bestEffort=True,
    )
    out = reduced.getInfo()  # returns dict with 'features'
    return out.get("features", [])

def safe_num(x):
    """Convert NaN/inf to None for JSON compliance, leave real numbers as is."""
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            if math.isfinite(x):
                return float(x)
            return None
        # strings etc.
        return x
    except Exception:
        return None

# ------------- Core computation -------------

def compute_latest_flags_for(gacc_list: Optional[List[str]]) -> Dict:
    init_ee()

    # 1) latest image with afg/pfg/HER bands
    img = latest_rap_image()

    # 2) PSAs (filtered by GACC if provided)
    geo = fetch_ca_psas_geojson()
    geo = filter_geojson_by_gacc(geo, gacc_list)

    # 3) zonal means from EE
    feats = ee_mean_by_polygons(img, geo)

    # 4) build dataframe of latest values
    rows = []
    for f in feats:
        props = f.get("properties", {}) or {}
        # IMPORTANT: read band keys directly (fix for "all zeros")
        her_val = props.get("HER")
        afg_val = props.get("afgNPP")
        pfg_val = props.get("pfgNPP")
        rows.append(
            {
                "PSANAME": props.get("PSANAME"),
                "GACCUnitID": props.get("GACCUnitID"),
                "afgNPP_latest": her_val if False else afg_val,  # keep both, set below
                "pfgNPP_latest": pfg_val,
                "HER_latest": her_val,
            }
        )

    latest_df = pd.DataFrame(rows)

    if latest_df.empty:
        return {"count": 0, "rows": []}

    # Ensure numeric
    for c in ["afgNPP_latest", "pfgNPP_latest", "HER_latest"]:
        latest_df[c] = pd.to_numeric(latest_df[c], errors="coerce")

    # 5) attach normals from CSV
    merged = latest_df.merge(
        NORMS_DF,
        on=["PSANAME", "GACCUnitID"],
        how="left",
        validate="m:1",
    )

    # 6) compute flag
    merged["above_normal"] = (merged["HER_latest"] > merged["HER_norm"]).fillna(False)

    # 7) prepare JSON-safe records
    records: List[Dict] = []
    for rec in merged.to_dict(orient="records"):
        out = {
            "PSANAME": rec.get("PSANAME"),
            "GACCUnitID": rec.get("GACCUnitID"),
            "afgNPP_latest": safe_num(rec.get("afgNPP_latest")),
            "pfgNPP_latest": safe_num(rec.get("pfgNPP_latest")),
            "HER_latest": safe_num(rec.get("HER_latest")),
            "afgNPP_norm": safe_num(rec.get("afgNPP_norm")),
            "pfgNPP_norm": safe_num(rec.get("pfgNPP_norm")),
            "HER_norm": safe_num(rec.get("HER_norm")),
            "above_normal": bool(rec.get("above_normal", False)),
        }
        records.append(out)

    return {"count": len(records), "rows": records}

# ------------- Routes -------------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "CA PSA Herbaceous API"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ee_initialized": EE_READY,
        "normals_csv": NORMS_CSV_PATH,
    }

@app.get("/psa_flags")
def psa_flags(
    gaccs: Optional[str] = Query(
        None,
        description="Comma-separated GACCUnitID values to include (e.g. USCAOSCC,USCAONCC). If omitted, returns all CA PSAs in CSV.",
    ),
    pretty: Optional[bool] = Query(
        False,
        description="If true, return pretty-formatted JSON.",
    ),
):
    try:
        gacc_list = [g.strip() for g in gaccs.split(",")] if gaccs else None
        payload = compute_latest_flags_for(gacc_list)

        if pretty:
            # pretty print while keeping a valid JSON object
            return JSONResponse(
                content=json.loads(json.dumps(payload, indent=2)),
                media_type="application/json",
            )
        return JSONResponse(content=payload)
    except Exception as e:
        log.exception("psa_flags failed")
        raise HTTPException(status_code=500, detail=str(e))

# ------------- Local dev entrypoint -------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
