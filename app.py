# app.py  v8
# CONUS PSA Herbaceous (HER) API — RAP 16-day provisional vs PSA normals CSV
#
# Env vars expected on Render:
#   EE_SERVICE_ACCOUNT      e.g. finefuel@finefuelloading.iam.gserviceaccount.com
#   EE_PRIVATE_KEY_FILE     e.g. /etc/secrets/ee-key.json
#   PSA_NORMALS_CSV         defaults to psa_HER_norm_CONUS_v1.csv
#   EE_COLLECTION_16D_PROV  defaults to …npp-partitioned-16day-v3-provisional
#   HTTP_TIMEOUT_SEC        defaults to 60
#   CORS_ALLOW_ORIGINS      defaults to "*"
#
# CHANGES v7 → v8:
#   1. FIX: get_psa_fc() now filters out NONE / "No PSA Assigned" junk rows
#   2. FIX: get_psa_fc() applies simplify(maxError=500) to reduce geometry
#           payload and prevent 502 timeouts on large GACCs (e.g. USGASAC)

from __future__ import annotations
import io
import json
import os
import threading
import time
import uuid
import datetime
from typing import List, Optional

import pandas as pd
import requests
import ee
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
PSA_FS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)
EE_COLLECTION_16D_PROV = os.getenv(
    "EE_COLLECTION_16D_PROV",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional",
)
EE_COLLECTION_16D_ARCH = os.getenv(
    "EE_COLLECTION_16D_ARCH",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3",
)
EE_SERVICE_ACCOUNT  = os.getenv("EE_SERVICE_ACCOUNT",  "").strip()
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "").strip()
NORMALS_CSV         = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CONUS_v1.csv")
HTTP_TIMEOUT_SEC    = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

# ─────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────
app = FastAPI(title="CONUS PSA Herbaceous API", version="8.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store for /generate_normals
JOBS: dict = {}
JOBS_LOCK = threading.Lock()

# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────
def safe_num(v):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    except Exception:
        return None


def parse_gaccs_param(gaccs_raw: Optional[str]) -> Optional[List[str]]:
    if not gaccs_raw:
        return None
    vals = [s.strip().upper() for s in gaccs_raw.split(",") if s.strip()]
    return vals or None


def is_junk_psa(psa_code: str) -> bool:
    """Return True for NONE / No PSA Assigned / empty codes."""
    c = psa_code.strip().upper()
    return not c or c == "NONE" or c.lower().startswith("no psa")


# ─────────────────────────────────────────
# Load normals CSV at startup
# ─────────────────────────────────────────
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
    df = df[~df["PSANationalCode"].astype(str).apply(is_junk_psa)]
    df["PSA_KEY"] = df["PSANationalCode"].astype(str).str.upper().str.strip()
    return df[["PSA_KEY", "PSANationalCode", "PSANAME", "GACCUnitID",
               "afgNPP_norm", "pfgNPP_norm", "HER_norm"]]


# ─────────────────────────────────────────
# Initialize EE once at startup
# ─────────────────────────────────────────
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

NORMALS_DF: Optional[pd.DataFrame] = None
try:
    NORMALS_DF = load_normals(NORMALS_CSV)
    print(f"Loaded normals: {len(NORMALS_DF)} rows from {NORMALS_CSV}")
except Exception as e:
    print("Normals load failed:", e)


# ─────────────────────────────────────────
# Fetch PSA polygons from ArcGIS (paginated)
# FIX v8: filters junk rows + simplifies geometries to prevent 502s
# ─────────────────────────────────────────
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
        psa_code = str(props.get("PSANationalCode", "")).strip()

        # FIX: skip junk rows (NONE / No PSA Assigned / empty)
        if not geom or is_junk_psa(psa_code):
            continue

        psa_key = psa_code.upper()

        # FIX: simplify geometry to prevent payload/timeout issues on large GACCs
        ee_geom = ee.Geometry(geom).simplify(maxError=500)

        ee_feats.append(ee.Feature(ee_geom, {
            "PSA_KEY":         psa_key,
            "PSANationalCode": psa_key,
            "PSANAME":         props.get("PSANAME"),
            "GACCUnitID":      props.get("GACCUnitID"),
        }))

    return ee.FeatureCollection(ee_feats)


# ─────────────────────────────────────────
# Core: compute latest flags vs normals
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Background worker for /generate_normals
# ─────────────────────────────────────────
def _run_generate_normals(job_id: str):
    def _update(msg: str):
        print(f"[{job_id}] {msg}")
        with JOBS_LOCK:
            JOBS[job_id]["message"] = msg

    try:
        _update("Starting — connecting to archived RAP collection (1986-2024)")
        arch   = ee.ImageCollection(EE_COLLECTION_16D_ARCH).filterDate("1986-01-01", "2025-01-01")
        n_imgs = arch.size().getInfo()
        if n_imgs == 0:
            raise RuntimeError(f"No images found in archived collection: {EE_COLLECTION_16D_ARCH}")
        _update(f"Archived collection has {n_imgs} composites — computing all-time mean")

        mean_bands = arch.select(["afgNPP", "pfgNPP"]).mean()
        her_band   = mean_bands.select("afgNPP").add(mean_bands.select("pfgNPP")).rename("HER")
        norm_stack = ee.Image.cat([mean_bands, her_band])

        _update("Fetching PSA polygons from ArcGIS ...")
        psa_fc = get_psa_fc(None)

        _update("Running reduceRegions (scale=90m, tileScale=16) ...")
        stats_fc = norm_stack.reduceRegions(
            collection=psa_fc,
            reducer=ee.Reducer.mean(),
            scale=90,
            tileScale=16,
        )
        stats = stats_fc.getInfo().get("features", [])
        _update(f"reduceRegions complete — {len(stats)} results — building CSV ...")

        rows = []
        for f in stats:
            p        = f.get("properties", {}) or {}
            psa_code = p.get("PSANationalCode", "")
            if not psa_code or is_junk_psa(psa_code):
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

        with JOBS_LOCK:
            JOBS[job_id]["status"]  = "complete"
            JOBS[job_id]["csv"]     = buf.getvalue()
            JOBS[job_id]["rows"]    = len(df)
            JOBS[job_id]["message"] = f"Done — {len(df)} PSAs"

    except Exception as exc:
        err = str(exc)
        print(f"[{job_id}] ERROR: {err}")
        with JOBS_LOCK:
            JOBS[job_id]["status"]  = "error"
            JOBS[job_id]["error"]   = err
            JOBS[job_id]["message"] = f"Failed: {err}"


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "CONUS PSA Herbaceous API v8\n\n"
        "Endpoints:\n"
        "  /psa_flags                  All CONUS\n"
        "  /psa_flags?gaccs=USCAONCC  Filter by GACC (comma-separated)\n"
        "  /psa_flags?pretty=1         Human-readable JSON\n"
        "  /generate_normals           Start background normals job\n"
        "  /normals_status?job_id=...  Poll job status / download CSV\n"
        "  /health                     Status check\n"
    )


@app.get("/health")
def health():
    with JOBS_LOCK:
        active_jobs = {
            jid: {"status": jdata["status"], "message": jdata["message"]}
            for jid, jdata in JOBS.items()
        }
    return {
        "status":         "ok",
        "ee_initialized": EE_READY,
        "normals_csv":    NORMALS_CSV,
        "normals_rows":   len(NORMALS_DF) if NORMALS_DF is not None else 0,
        "active_jobs":    active_jobs,
    }


@app.get("/psa_flags")
def psa_flags(
    gaccs:  Optional[str] = Query(None),
    pretty: Optional[int] = Query(0),
):
    try:
        gacc_list = parse_gaccs_param(gaccs)
        payload   = compute_latest_flags(gacc_list)
        if bool(pretty):
            return PlainTextResponse(json.dumps(payload, indent=2), media_type="application/json")
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate_normals")
def generate_normals(background_tasks: BackgroundTasks):
    if not EE_READY:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized.")
    job_id = str(uuid.uuid4())[:8]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status":  "running",
            "started": datetime.datetime.utcnow().isoformat() + "Z",
            "message": "Job queued",
            "csv":     None,
            "rows":    0,
            "error":   None,
        }
    background_tasks.add_task(_run_generate_normals, job_id)
    return JSONResponse({
        "job_id":   job_id,
        "status":   "running",
        "poll_url": f"/normals_status?job_id={job_id}",
    })


@app.get("/normals_status")
def normals_status(job_id: str = Query(...)):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] == "complete":
        return PlainTextResponse(
            job["csv"],
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=psa_HER_norm_CONUS_v1.csv"},
        )
    return JSONResponse({
        "job_id":  job_id,
        "status":  job["status"],
        "started": job["started"],
        "message": job["message"],
        "error":   job.get("error"),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
