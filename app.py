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
# CHANGES FROM v4 → v6:
#   1. FIX: above_normal — null-safe; PSAs missing normals return null not 0
#   2. FIX: load_normals() filters out "No PSA Assigned" junk rows
#   3. FIX: get_psa_fc() paginates ArcGIS (handles exceededTransferLimit)
#   4. FIX: tileScale scales automatically — 4/8/16 based on query size
#   5. FIX: above_normal flag uses correct null-safe comparison
#   6. NEW: Default CSV → psa_HER_norm_CONUS_v1.csv
#   7. NEW: gaccs in response lists actual GACCs found in results
#   8. NEW: /health reports normals row count
#   9. NEW: /generate_normals endpoint — computes CONUS normals CSV via GEE,
#           returns as downloadable CSV. Run once then redeploy with the file.

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
EE_SERVICE_ACCOUNT  = os.getenv("EE_SERVICE_ACCOUNT",  "").strip()
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "").strip()
NORMALS_CSV         = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CONUS_v1.csv")
HTTP_TIMEOUT_SEC    = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="CONUS PSA Herbaceous API", version="6.0.0")
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
        "afgNPP_norm", "pfgNPP_norm", "HER_norm",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Normals CSV missing column: {col}")

    # Drop junk rows — blank codes and "No PSA Assigned" variants
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
# Fetch PSA polygons from ArcGIS
# Paginates through exceededTransferLimit for national queries
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
            "f":                 "geojson",
            "outSR":             4326,
            "returnGeometry":    "true",
            "outFields":         "PSANationalCode,PSANAME,GACCUnitID",
            "where":             where,
            "resultOffset":      offset,
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

    # Latest RAP 16-day composite
    coll        = ee.ImageCollection(EE_COLLECTION_16D_PROV).sort("system:time_start", False)
    latest      = coll.first()
    latest_date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    # Bands + HER
    img   = latest.select(["afgNPP", "pfgNPP"])
    her   = img.select("afgNPP").add(img.select("pfgNPP")).rename("HER")
    stack = img.addBands(her)

    # PSA polygons
    psa_fc = get_psa_fc(gaccs_list)

    # tileScale: 4 = single GACC, 8 = 2-4 GACCs, 16 = 5+ or full CONUS
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
            "count":            0,
            "gaccs":            gaccs_list or [],
            "collection":       EE_COLLECTION_16D_PROV,
            "latest_composite": latest_date,
            "rows":             [],
        }

    # Merge with normals
    merged = pd.merge(
        latest_df, NORMALS_DF,
        on="PSA_KEY", how="left",
        suffixes=("", "_norms"),
    )

    # Null-safe above_normal flag
    # 1 = above normal | 0 = below/at normal | None = no normals data for this PSA
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
        "CONUS PSA Herbaceous API v6 — /psa_flags?pretty=1 for all CONUS\n"
        "Filter by GACC: /psa_flags?gaccs=USCAOSCC,USCAONCC&pretty=1\n\n"
        "Available GACCs:\n"
        "  USCAONCC  Northern California\n"
        "  USCAOSCC  Southern California\n"
        "  USGBONCC  Great Basin\n"
        "  USNMSWCC  Southwest\n"
        "  USRMSRCC  Rocky Mountain South\n"
        "  USRMSNCC  Rocky Mountain North\n"
        "  USNRFNCC  Northwest\n"
        "  USSASWCC  Southern Area\n"
        "  USEASNCC  Eastern Area\n"
        "  USNRENCC  Northeast\n\n"
        "One-time setup: /generate_normals  (generates CONUS normals CSV via GEE)\n"
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
    gaccs:   Optional[str] = Query(None, description="Comma-separated GACCUnitID values; omit for all CONUS"),
    pretty:  Optional[int] = Query(0,    description="Pretty-print JSON if 1"),
    pretty1: Optional[int] = Query(0,    description="Alias for pretty=1"),
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
def generate_normals(
    baseline_start: int = Query(2017, description="First year of baseline period"),
    baseline_end:   int = Query(2022, description="Last year of baseline period"),
    doy_window:     int = Query(8,    description="Days +/- around target DOY to search"),
):
    """
    ONE-TIME SETUP ENDPOINT.

    Computes CONUS PSA normals from the RAP 16-day provisional collection
    using the GEE credentials already configured on this server.

    Steps:
      1. Hit this URL in your browser — it will take 10-15 minutes
      2. Your browser will download psa_HER_norm_CONUS_v1.csv automatically
      3. Add that CSV to your repo and redeploy
      4. Update PSA_NORMALS_CSV env var to psa_HER_norm_CONUS_v1.csv
      5. All CONUS PSAs will then return correct above_normal values

    Optional parameters:
      ?baseline_start=2017&baseline_end=2022&doy_window=8
    """
    if not EE_READY:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized.")

    # Determine target DOY from latest composite
    coll       = ee.ImageCollection(EE_COLLECTION_16D_PROV).sort("system:time_start", False)
    latest     = coll.first()
    latest_ms  = latest.get("system:time_start").getInfo()
    latest_dt  = datetime.datetime.utcfromtimestamp(latest_ms / 1000)
    target_doy = latest_dt.timetuple().tm_yday
    print(f"generate_normals: target DOY {target_doy} ({latest_dt.date()}), "
          f"baseline {baseline_start}–{baseline_end}, window ±{doy_window} days")

    # Collect one matching composite per baseline year
    baseline_imgs = []
    for year in range(baseline_start, baseline_end + 1):
        base_date    = datetime.date(year, 1, 1) + datetime.timedelta(days=target_doy - 1)
        window_start = (base_date - datetime.timedelta(days=doy_window)).isoformat()
        window_end   = (base_date + datetime.timedelta(days=doy_window + 1)).isoformat()
        year_coll    = coll.filterDate(window_start, window_end)
        count        = year_coll.size().getInfo()
        if count > 0:
            baseline_imgs.append(year_coll.sort("system:time_start").first())
            print(f"  {year}: composite found in window")
        else:
            print(f"  {year}: no composite — skipping")

    if not baseline_imgs:
        raise HTTPException(status_code=500, detail="No baseline composites found for specified period.")

    # Average baseline composites
    mean_img   = ee.ImageCollection(baseline_imgs).select(["afgNPP", "pfgNPP"]).mean()
    her_band   = mean_img.select("afgNPP").add(mean_img.select("pfgNPP")).rename("HER")
    norm_stack = mean_img.addBands(her_band)

    # All CONUS PSA polygons
    psa_fc = get_psa_fc(None)

    print("generate_normals: running reduceRegions for all CONUS PSAs ...")
    stats_fc = norm_stack.reduceRegions(
        collection=psa_fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=16,
    )
    stats = stats_fc.getInfo().get("features", [])
    print(f"generate_normals: {len(stats)} PSA results returned")

    # Build CSV
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

    print(f"generate_normals: returning CSV with {len(df)} rows")
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
