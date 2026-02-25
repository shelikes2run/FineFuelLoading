# app.py
# CONUS PSA Herbaceous (HER) API — RAP 16-day provisional (afgNPP/pfgNPP) vs PSA normals CSV
#
# Env vars expected on Render (or local):
#   EE_SERVICE_ACCOUNT      e.g. finefuel@finefuelloading.iam.gserviceaccount.com
#   EE_PRIVATE_KEY_FILE     e.g. /etc/secrets/ee-key.json
#   PSA_NORMALS_CSV         defaults to psa_HER_norm_CONUS_v1.csv
#   EE_COLLECTION_16D_PROV  defaults to projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional
#   HTTP_TIMEOUT_SEC        defaults to 60
#   CORS_ALLOW_ORIGINS      defaults to "*"
#
# CHANGES v4 → v7:
#   1. FIX: above_normal null-safe (fillna(1e9) bug removed)
#   2. FIX: load_normals() strips junk "No PSA Assigned" rows
#   3. FIX: get_psa_fc() paginates ArcGIS (exceededTransferLimit)
#   4. FIX: tileScale auto-scales 4/8/16 by query size
#   5. NEW: Default CSV → psa_HER_norm_CONUS_v1.csv
#   6. NEW: gaccs in response lists actual GACCs in results
#   7. NEW: /health reports normals row count
#   8. FIX: /generate_normals — switched to ARCHIVED collection (1986–2024),
#           all-time mean (no DOY filtering), scale=90m, geometry simplify,
#           tileScale=16 for CONUS

from __future__ import annotations
import os
import io
import json
import datetime
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
# Archived collection used for normals generation (1986–2024 long-term mean)
EE_COLLECTION_16D_ARCH = os.getenv(
    "EE_COLLECTION_16D_ARCH",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3",
)
EE_SERVICE_ACCOUNT  = os.getenv("EE_SERVICE_ACCOUNT",  "").strip()
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "").strip()
NORMALS_CSV         = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CONUS_v1.csv")
HTTP_TIMEOUT_SEC    = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="CONUS PSA Herbaceous API", version="7.0.0")
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
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:
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
# Load normals CSV at startup
# ----------------------------
def load_normals(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normals CSV not found: {path}")
    df = pd.read_csv(path)
    required = ["PSANationalCode", "PSANAME", "GACCUnitID",
                "afgNPP_norm", "pfgNPP_norm", "HER_norm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Normals CSV missing column: {col}")
    df = df[df["PSANationalCode"].notna()]
    df = df[df["PSANationalCode"].astype(str).str.strip() != ""]
    df = df[~df["PSANationalCode"].astype(str).str.lower().str.startswith("no psa")]
    df["PSA_KEY"] = df["PSANationalCode"].astype(str).str.upper().str.strip()
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

# Load normals once at startup
NORMALS_DF: Optional[pd.DataFrame] = None
try:
    NORMALS_DF = load_normals(NORMALS_CSV)
    print(f"Loaded normals: {len(NORMALS_DF)} rows from {NORMALS_CSV}")
except Exception as e:
    print("Normals load failed:", e)


# ----------------------------
# Fetch PSA polygons from ArcGIS (paginated)
# ----------------------------
def get_psa_fc(gaccs: Optional[List[str]]) -> ee.FeatureCollection:
    where = "1=1"
    if gaccs:
        quoted = ",".join([f"'{g}'" for g in gaccs])
        where  = f"GACCUnitID IN ({quoted})"

    all_features: list = []
    offset    = 0
    page_size = 1000

    while True:
        params = {
            "f": "geojson", "outSR": 4326, "returnGeometry": "true",
            "outFields": "PSANationalCode,PSANAME,GACCUnitID",
            "where": where,
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        r = requests.get(PSA_FS_URL, params=params, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        gj   = r.json()
        page = gj.get("features", [])
        all_features.extend(page)
        if not gj.get("exceededTransferLimit", False) or not page:
            break
        offset += len(page)

    ee_feats = []
    for f in all_features:
        props    = f.get("properties", {}) or {}
        geom     = f.get("geometry")
        if not geom:
            continue
        psa_code = str(props.get("PSANationalCode", "")).upper().strip()
        if not psa_code:
            continue
        ee_feats.append(ee.Feature(ee.Geometry(geom), {
            "PSA_KEY":         psa_code,
            "PSANationalCode": psa_code,
            "PSANAME":         props.get("PSANAME"),
            "GACCUnitID":      props.get("GACCUnitID"),
        }))

    return ee.FeatureCollection(ee_feats)


# ----------------------------
# Core: compute latest flags vs normals
# ----------------------------
def compute_latest_flags(gaccs_list: Optional[List[str]]):
    if not EE_READY:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized on server.")
    if NORMALS_DF is None or NORMALS_DF.empty:
        raise HTTPException(status_code=500, detail="Normals table not loaded.")

    coll        = ee.ImageCollection(EE_COLLECTION_16D_PROV).sort("system:time_start", False)
    latest      = coll.first()
    latest_date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    img   = latest.select(["afgNPP", "pfgNPP"])
    her   = img.select("afgNPP").add(img.select("pfgNPP")).rename("HER")
    stack = img.addBands(her)

    psa_fc = get_psa_fc(gaccs_list)

    if not gaccs_list:
        tile_scale = 16
    elif len(gaccs_list) >= 5:
        tile_scale = 16
    elif len(gaccs_list) >= 2:
        tile_scale = 8
    else:
        tile_scale = 4

    stats_fc = stack.reduceRegions(
        collection=psa_fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=tile_scale,
    )
    stats = stats_fc.getInfo().get("features", [])

    latest_rows = []
    for f in stats:
        p = f.get("properties", {}) or {}
        latest_rows.append({
            "PSA_KEY":         p.get("PSA_KEY"),
            "PSANationalCode": p.get("PSANationalCode"),
            "PSANAME":         p.get("PSANAME"),
            "GACCUnitID":      p.get("GACCUnitID"),
            "afgNPP_latest":   safe_num(p.get("afgNPP")),
            "pfgNPP_latest":   safe_num(p.get("pfgNPP")),
            "HER_latest":      safe_num(p.get("HER")),
        })
    latest_df = pd.DataFrame(latest_rows)

    if latest_df.empty:
        return {
            "count": 0, "gaccs": gaccs_list or [],
            "collection": EE_COLLECTION_16D_PROV,
            "latest_composite": latest_date, "rows": [],
        }

    merged = pd.merge(latest_df, NORMALS_DF, on="PSA_KEY", how="left", suffixes=("", "_norms"))

    merged["above_normal"] = pd.NA
    valid = merged["HER_latest"].notna() & merged["HER_norm"].notna()
    merged.loc[valid, "above_normal"] = (
        merged.loc[valid, "HER_latest"] > merged.loc[valid, "HER_norm"]
    ).astype(int)

    result_gaccs = sorted(merged["GACCUnitID"].dropna().unique().tolist())

    out = []
    for _, r in merged.iterrows():
        an = r.get("above_normal")
        out.append({
            "PSA":           r.get("PSANationalCode"),
            "PSANAME":       r.get("PSANAME"),
            "GACCUnitID":    r.get("GACCUnitID"),
            "afgNPP_latest": safe_num(r.get("afgNPP_latest")),
            "pfgNPP_latest": safe_num(r.get("pfgNPP_latest")),
            "HER_latest":    safe_num(r.get("HER_latest")),
            "afgNPP_norm":   safe_num(r.get("afgNPP_norm")),
            "pfgNPP_norm":   safe_num(r.get("pfgNPP_norm")),
            "HER_norm":      safe_num(r.get("HER_norm")),
            "above_normal":  int(an) if pd.notna(an) else None,
        })

    return {
        "count":            len(out),
        "gaccs":            gaccs_list if gaccs_list else result_gaccs,
        "collection":       EE_COLLECTION_16D_PROV,
        "latest_composite": latest_date,
        "rows":             out,
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "CONUS PSA Herbaceous API v7\n\n"
        "Endpoints:\n"
        "  /psa_flags                  All CONUS (omit gaccs param)\n"
        "  /psa_flags?gaccs=USCAONCC  Filter by GACC\n"
        "  /psa_flags?pretty=1         Human-readable JSON\n"
        "  /generate_normals           One-time: build CONUS normals CSV\n"
        "  /health                     Status check\n\n"
        "GACC codes:\n"
        "  USCAONCC  Northern California\n"
        "  USCAOSCC  Southern California\n"
        "  USGBONCC  Great Basin\n"
        "  USNMSWCC  Southwest\n"
        "  USRMSRCC  Rocky Mountain South\n"
        "  USRMSNCC  Rocky Mountain North\n"
        "  USNRFNCC  Northwest\n"
        "  USSASWCC  Southern Area\n"
        "  USEASNCC  Eastern Area\n"
        "  USNRENCC  Northeast\n"
    )


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "ee_initialized": EE_READY,
        "normals_csv":    NORMALS_CSV,
        "normals_rows":   len(NORMALS_DF) if NORMALS_DF is not None else 0,
    }


@app.get("/psa_flags")
def psa_flags(
    gaccs:   Optional[str] = Query(None),
    pretty:  Optional[int] = Query(0),
    pretty1: Optional[int] = Query(0),
):
    try:
        gacc_list = parse_gaccs_param(gaccs)
        payload   = compute_latest_flags(gacc_list)
        if bool(pretty) or bool(pretty1):
            return PlainTextResponse(json.dumps(payload, indent=2), media_type="application/json")
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate_normals")
def generate_normals():
    """
    ONE-TIME SETUP — generates psa_HER_norm_CONUS_v1.csv.

    Uses the ARCHIVED RAP 16-day collection (1986–2024) and computes the
    all-time mean of afgNPP and pfgNPP across all available composites.
    This matches the method used to generate the original CA normals.

    - Collection : projects/rap-data-365417/assets/npp-partitioned-16day-v3
    - Date range : 1986-01-01 → 2025-01-01 (all available composites)
    - Stat       : mean of ALL composites (no DOY filtering)
    - Scale      : 90 m  (matches original CA generation)
    - tileScale  : 16    (required for CONUS scale)
    - Geometry   : simplify(maxError=120) per polygon

    Hit this URL once (expect 15-30 min) then commit the downloaded CSV
    to your repo as psa_HER_norm_CONUS_v1.csv and redeploy.
    """
    if not EE_READY:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized.")

    print("generate_normals: starting — archived collection, all-time mean, 90m scale")

    # ── Step 1: Build the all-time mean image from the archived collection ──
    arch = ee.ImageCollection(EE_COLLECTION_16D_ARCH).filterDate("1986-01-01", "2025-01-01")

    # Confirm images exist
    n_imgs = arch.size().getInfo()
    if n_imgs == 0:
        raise HTTPException(
            status_code=500,
            detail=f"No images found in archived collection {EE_COLLECTION_16D_ARCH}."
        )
    print(f"generate_normals: {n_imgs} composites found in archived collection")

    mean_bands = arch.select(["afgNPP", "pfgNPP"]).mean()
    her_band   = mean_bands.select("afgNPP").add(mean_bands.select("pfgNPP")).rename("HER")
    norm_stack = ee.Image.cat([mean_bands, her_band])   # bands: afgNPP, pfgNPP, HER

    # ── Step 2: Fetch all CONUS PSA polygons (paginated), with simplification ──
    print("generate_normals: fetching PSA polygons from ArcGIS ...")
    all_features: list = []
    offset    = 0
    page_size = 1000

    while True:
        params = {
            "f": "geojson", "outSR": 4326, "returnGeometry": "true",
            "outFields": "PSANationalCode,PSANAME,GACCUnitID",
            "where": "1=1",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        r = requests.get(PSA_FS_URL, params=params, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        gj   = r.json()
        page = gj.get("features", [])
        all_features.extend(page)
        if not gj.get("exceededTransferLimit", False) or not page:
            break
        offset += len(page)

    ee_feats = []
    skipped  = 0
    for f in all_features:
        props    = f.get("properties", {}) or {}
        geom_raw = f.get("geometry")
        psa_code = str(props.get("PSANationalCode", "")).upper().strip()
        if not geom_raw or not psa_code or psa_code.lower().startswith("no psa"):
            skipped += 1
            continue
        # Simplify geometry to reduce GEE payload size (matches original 90m.ipynb)
        geom = ee.Geometry(geom_raw).simplify(maxError=120)
        ee_feats.append(ee.Feature(geom, {
            "PSA_KEY":         psa_code,
            "PSANationalCode": psa_code,
            "PSANAME":         props.get("PSANAME", ""),
            "GACCUnitID":      props.get("GACCUnitID", ""),
        }))

    print(f"generate_normals: {len(ee_feats)} valid PSA polygons ({skipped} skipped)")
    if not ee_feats:
        raise HTTPException(status_code=500, detail="No valid PSA polygons returned from ArcGIS.")

    psa_fc = ee.FeatureCollection(ee_feats)

    # ── Step 3: reduceRegions at 90m ──
    print("generate_normals: running reduceRegions (scale=90m, tileScale=16) — this may take 15-30 min ...")
    stats_fc = norm_stack.reduceRegions(
        collection=psa_fc,
        reducer=ee.Reducer.mean(),
        scale=90,
        tileScale=16,
    )
    stats = stats_fc.getInfo().get("features", [])
    print(f"generate_normals: {len(stats)} PSA results returned")

    # ── Step 4: Build CSV ──
    rows = []
    for f in stats:
        p        = f.get("properties", {}) or {}
        psa_code = p.get("PSANationalCode", "")
        if not psa_code:
            continue
        rows.append({
            "PSANationalCode": psa_code,
            "PSANAME":         p.get("PSANAME",    ""),
            "GACCUnitID":      p.get("GACCUnitID", ""),
            "afgNPP_norm":     safe_num(p.get("afgNPP")),
            "pfgNPP_norm":     safe_num(p.get("pfgNPP")),
            "HER_norm":        safe_num(p.get("HER")),
        })

    df  = pd.DataFrame(rows).sort_values(["GACCUnitID", "PSANationalCode"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)

    null_count = df["HER_norm"].isna().sum()
    print(
        f"generate_normals: done — {len(df)} PSAs "
        f"({null_count} with null HER_norm — check geometry/RAP coverage)"
    )

    return PlainTextResponse(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=psa_HER_norm_CONUS_v1.csv"},
    )


# ----------------------------
# Local run
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
