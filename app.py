# app.py  v10
# CONUS PSA Herbaceous (HER) API — RAP 16-day provisional vs PSA normals CSV
#
# Env vars expected on Render:
#   EE_SERVICE_ACCOUNT      e.g. finefuel@finefuelloading.iam.gserviceaccount.com
#   EE_PRIVATE_KEY_FILE     e.g. /etc/secrets/ee-key.json
#   PSA_NORMALS_CSV         defaults to psa_HER_norm_CONUS_v1.csv
#   EE_COLLECTION_16D_PROV  defaults to …npp-partitioned-16day-v3-provisional
#   CACHE_REFRESH_HOURS     how often to refresh the cache (default 6)
#   HTTP_TIMEOUT_SEC        defaults to 60
#   CORS_ALLOW_ORIGINS      defaults to "*"
#
# CHANGES v9 → v10:
#   KEY FIX: process one GACC at a time in the background cache loop instead
#   of all 240 CONUS PSAs in a single reduceRegions call.
#   This keeps peak memory well under 512MB, eliminating the OOM crash on Render.
#   Each GACC is fetched, reduced, and released before the next begins.
#   If one GACC fails (GEE timeout etc.) the rest still complete.

from __future__ import annotations
import io, json, os, threading, time, uuid, datetime
from typing import List, Optional

import pandas as pd
import requests
import ee
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────
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
EE_SERVICE_ACCOUNT   = os.getenv("EE_SERVICE_ACCOUNT",  "").strip()
EE_PRIVATE_KEY_FILE  = os.getenv("EE_PRIVATE_KEY_FILE", "").strip()
NORMALS_CSV          = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CONUS_v1.csv")
HTTP_TIMEOUT_SEC     = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))
CACHE_REFRESH_HOURS  = float(os.getenv("CACHE_REFRESH_HOURS", "6"))

# All CONUS GACCs — processed one at a time to keep memory low
GACC_CODES = [
    "USAKACC",
    "USCAONCC",
    "USCAOSCC",
    "USCORMC",
    "USGASAC",
    "USMTNRC",
    "USNMSWC",
    "USORNWC",
    "USUTGBC",
    "USWIEACC",
]

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="CONUS PSA Herbaceous API", version="10.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ── In-memory cache ───────────────────────────────────────────────────────────
_CACHE: dict = {
    "rows":             [],
    "latest_composite": None,
    "computed_at":      None,
    "status":           "starting",
    "error":            None,
    "gacc_progress":    {},
}
_CACHE_LOCK = threading.Lock()

JOBS: dict = {}
JOBS_LOCK = threading.Lock()

# ── Utilities ─────────────────────────────────────────────────────────────────
def safe_num(v):
    try:
        if v is None: return None
        f = float(v)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    except Exception: return None

def parse_gaccs_param(gaccs_raw: Optional[str]) -> Optional[List[str]]:
    if not gaccs_raw: return None
    return [s.strip().upper() for s in gaccs_raw.split(",") if s.strip()] or None

def is_junk_psa(psa_code: str) -> bool:
    c = str(psa_code).strip().upper()
    return not c or c == "NONE" or c.lower().startswith("no psa")

# ── Load normals CSV ──────────────────────────────────────────────────────────
def load_normals(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normals CSV not found: {path}")
    df = pd.read_csv(path)
    required = ["PSANationalCode","PSANAME","GACCUnitID",
                "afgNPP_norm","pfgNPP_norm","HER_norm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Normals CSV missing column: {col}")
    df = df[df["PSANationalCode"].notna()]
    df = df[~df["PSANationalCode"].astype(str).apply(is_junk_psa)]
    df["PSA_KEY"] = df["PSANationalCode"].astype(str).str.upper().str.strip()
    return df[["PSA_KEY","PSANationalCode","PSANAME","GACCUnitID",
               "afgNPP_norm","pfgNPP_norm","HER_norm"]]

# ── Init EE ───────────────────────────────────────────────────────────────────
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

# ── Fetch PSA polygons ────────────────────────────────────────────────────────
def get_psa_fc(gaccs: Optional[List[str]] = None) -> ee.FeatureCollection:
    where = "1=1"
    if gaccs:
        quoted = ",".join([f"'{g}'" for g in gaccs])
        where  = f"GACCUnitID IN ({quoted})"

    all_features: list = []
    offset = 0
    while True:
        params = {
            "f": "geojson", "outSR": 4326, "returnGeometry": "true",
            "outFields": "PSANationalCode,PSANAME,GACCUnitID",
            "where": where, "resultOffset": offset, "resultRecordCount": 1000,
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
        if not geom or is_junk_psa(psa_code):
            continue
        psa_key = psa_code.upper()
        ee_geom = ee.Geometry(geom).simplify(maxError=500)
        ee_feats.append(ee.Feature(ee_geom, {
            "PSA_KEY":         psa_key,
            "PSANationalCode": psa_key,
            "PSANAME":         props.get("PSANAME"),
            "GACCUnitID":      props.get("GACCUnitID"),
        }))
    return ee.FeatureCollection(ee_feats)

# ── Core compute (background thread only) ────────────────────────────────────
def _compute_all_conus():
    print("Cache: starting CONUS computation (per-GACC batching) ...")
    try:
        if not EE_READY:
            raise RuntimeError("Earth Engine not initialized")
        if NORMALS_DF is None or NORMALS_DF.empty:
            raise RuntimeError("Normals table not loaded")

        coll        = ee.ImageCollection(EE_COLLECTION_16D_PROV).sort("system:time_start", False)
        latest      = coll.first()
        latest_date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        print(f"Cache: latest composite = {latest_date}")

        img   = latest.select(["afgNPP","pfgNPP"])
        her   = img.select("afgNPP").add(img.select("pfgNPP")).rename("HER")
        stack = img.addBands(her)

        all_latest_rows = []
        gacc_progress   = {}

        for gacc in GACC_CODES:
            print(f"Cache: processing {gacc} ...")
            try:
                psa_fc  = get_psa_fc([gacc])
                fc_size = psa_fc.size().getInfo()

                if fc_size == 0:
                    print(f"Cache: {gacc} — 0 valid PSAs, skipping")
                    gacc_progress[gacc] = {"status": "skipped", "psas": 0}
                    continue

                stats_fc = stack.reduceRegions(
                    collection=psa_fc,
                    reducer=ee.Reducer.mean(),
                    scale=90,
                    tileScale=16,
                )
                stats = stats_fc.getInfo().get("features", [])
                print(f"Cache: {gacc} — {len(stats)} results")

                for f in stats:
                    p = f.get("properties", {}) or {}
                    all_latest_rows.append({
                        "PSA_KEY":         p.get("PSA_KEY"),
                        "PSANationalCode": p.get("PSANationalCode"),
                        "PSANAME":         p.get("PSANAME"),
                        "GACCUnitID":      p.get("GACCUnitID"),
                        "afgNPP_latest":   safe_num(p.get("afgNPP")),
                        "pfgNPP_latest":   safe_num(p.get("pfgNPP")),
                        "HER_latest":      safe_num(p.get("HER")),
                    })
                gacc_progress[gacc] = {"status": "ok", "psas": len(stats)}

            except Exception as gacc_exc:
                print(f"Cache: {gacc} failed — {gacc_exc}")
                gacc_progress[gacc] = {"status": "error", "error": str(gacc_exc)}
                continue

        latest_df = pd.DataFrame(all_latest_rows)
        if latest_df.empty:
            raise RuntimeError("No PSA results returned from any GACC")

        merged = pd.merge(latest_df, NORMALS_DF, on="PSA_KEY", how="left", suffixes=("","_norms"))

        merged["above_normal"] = pd.NA
        valid = merged["HER_latest"].notna() & merged["HER_norm"].notna()
        merged.loc[valid, "above_normal"] = (
            merged.loc[valid, "HER_latest"] > merged.loc[valid, "HER_norm"]
        ).astype(int)

        rows = []
        for _, r in merged.iterrows():
            an = r.get("above_normal")
            rows.append({
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

        with _CACHE_LOCK:
            _CACHE["rows"]             = rows
            _CACHE["latest_composite"] = latest_date
            _CACHE["computed_at"]      = datetime.datetime.utcnow().isoformat() + "Z"
            _CACHE["status"]           = "ready"
            _CACHE["error"]            = None
            _CACHE["gacc_progress"]    = gacc_progress

        print(f"Cache: ready — {len(rows)} PSA rows stored across {len(gacc_progress)} GACCs")

    except Exception as exc:
        print(f"Cache: compute failed — {exc}")
        with _CACHE_LOCK:
            _CACHE["status"] = "error"
            _CACHE["error"]  = str(exc)


def _cache_refresh_loop():
    while True:
        _compute_all_conus()
        sleep_secs = CACHE_REFRESH_HOURS * 3600
        print(f"Cache: next refresh in {CACHE_REFRESH_HOURS}h")
        time.sleep(sleep_secs)


if EE_READY and NORMALS_DF is not None:
    t = threading.Thread(target=_cache_refresh_loop, daemon=True)
    t.start()
    print("Cache: background refresh thread started")
else:
    print("Cache: skipping background thread (EE or normals not ready)")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "CONUS PSA Herbaceous API v10\n\n"
        "Endpoints:\n"
        "  /psa_flags                  All CONUS (served from cache)\n"
        "  /psa_flags?gaccs=USCAONCC  Filter by GACC (comma-separated)\n"
        "  /psa_flags?pretty=1         Human-readable JSON\n"
        "  /health                     Cache status + row count + per-GACC progress\n"
        "  /generate_normals           Start one-time normals background job\n"
        "  /normals_status?job_id=...  Poll job / download CSV\n"
    )


@app.get("/health")
def health():
    with _CACHE_LOCK:
        cache_status    = _CACHE["status"]
        cache_rows      = len(_CACHE["rows"])
        cache_computed  = _CACHE["computed_at"]
        cache_composite = _CACHE["latest_composite"]
        cache_error     = _CACHE["error"]
        gacc_progress   = dict(_CACHE["gacc_progress"])
    with JOBS_LOCK:
        active_jobs = {jid:{"status":j["status"],"message":j["message"]} for jid,j in JOBS.items()}
    return {
        "status":           "ok",
        "ee_initialized":   EE_READY,
        "normals_csv":      NORMALS_CSV,
        "normals_rows":     len(NORMALS_DF) if NORMALS_DF is not None else 0,
        "cache_status":     cache_status,
        "cache_psa_rows":   cache_rows,
        "cache_computed_at":cache_computed,
        "latest_composite": cache_composite,
        "cache_error":      cache_error,
        "gacc_progress":    gacc_progress,
        "active_jobs":      active_jobs,
    }


@app.get("/psa_flags")
def psa_flags(
    gaccs:  Optional[str] = Query(None),
    pretty: Optional[int] = Query(0),
):
    with _CACHE_LOCK:
        status   = _CACHE["status"]
        rows     = list(_CACHE["rows"])
        latest   = _CACHE["latest_composite"]
        computed = _CACHE["computed_at"]

    if status == "starting":
        raise HTTPException(status_code=503, detail="Cache is still computing — try again in a few minutes.")
    if status == "error":
        raise HTTPException(status_code=500, detail=f"Cache compute failed: {_CACHE['error']}")

    gacc_list = parse_gaccs_param(gaccs)
    if gacc_list:
        rows = [r for r in rows if r.get("GACCUnitID") in gacc_list]

    result_gaccs = sorted(set(r["GACCUnitID"] for r in rows if r.get("GACCUnitID")))

    payload = {
        "count":            len(rows),
        "gaccs":            gacc_list if gacc_list else result_gaccs,
        "collection":       EE_COLLECTION_16D_PROV,
        "latest_composite": latest,
        "cache_computed_at":computed,
        "rows":             rows,
    }

    if bool(pretty):
        return PlainTextResponse(json.dumps(payload, indent=2), media_type="application/json")
    return JSONResponse(payload)


# ── Background normals job ────────────────────────────────────────────────────
def _run_generate_normals(job_id: str):
    def _update(msg):
        print(f"[{job_id}] {msg}")
        with JOBS_LOCK:
            JOBS[job_id]["message"] = msg
    try:
        _update("Connecting to archived collection")
        arch   = ee.ImageCollection(EE_COLLECTION_16D_ARCH).filterDate("1986-01-01","2025-01-01")
        n_imgs = arch.size().getInfo()
        if n_imgs == 0:
            raise RuntimeError("No images in archived collection")
        _update(f"{n_imgs} composites — computing mean")
        mean_bands = arch.select(["afgNPP","pfgNPP"]).mean()
        her_band   = mean_bands.select("afgNPP").add(mean_bands.select("pfgNPP")).rename("HER")
        norm_stack = ee.Image.cat([mean_bands, her_band])
        _update("Fetching PSA polygons")
        psa_fc = get_psa_fc(None)
        _update("Running reduceRegions")
        stats_fc = norm_stack.reduceRegions(collection=psa_fc, reducer=ee.Reducer.mean(), scale=90, tileScale=16)
        stats = stats_fc.getInfo().get("features", [])
        rows = []
        for f in stats:
            p = f.get("properties",{}) or {}
            psa_code = p.get("PSANationalCode","")
            if not psa_code or is_junk_psa(psa_code): continue
            rows.append({"PSANationalCode":psa_code,"PSANAME":p.get("PSANAME",""),"GACCUnitID":p.get("GACCUnitID",""),
                         "afgNPP_norm":safe_num(p.get("afgNPP")),"pfgNPP_norm":safe_num(p.get("pfgNPP")),"HER_norm":safe_num(p.get("HER"))})
        df  = pd.DataFrame(rows).sort_values(["GACCUnitID","PSANationalCode"])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        with JOBS_LOCK:
            JOBS[job_id].update({"status":"complete","csv":buf.getvalue(),"rows":len(df),"message":f"Done — {len(df)} PSAs"})
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update({"status":"error","error":str(exc),"message":f"Failed: {exc}"})


@app.get("/generate_normals")
def generate_normals(background_tasks: BackgroundTasks):
    if not EE_READY:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized.")
    job_id = str(uuid.uuid4())[:8]
    with JOBS_LOCK:
        JOBS[job_id] = {"status":"running","started":datetime.datetime.utcnow().isoformat()+"Z",
                        "message":"Job queued","csv":None,"rows":0,"error":None}
    background_tasks.add_task(_run_generate_normals, job_id)
    return JSONResponse({"job_id":job_id,"status":"running","poll_url":f"/normals_status?job_id={job_id}"})


@app.get("/normals_status")
def normals_status(job_id: str = Query(...)):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] == "complete":
        return PlainTextResponse(job["csv"], media_type="text/csv",
            headers={"Content-Disposition":"attachment; filename=psa_HER_norm_CONUS_v1.csv"})
    return JSONResponse({"job_id":job_id,"status":job["status"],"started":job["started"],
                         "message":job["message"],"error":job.get("error")})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")))
