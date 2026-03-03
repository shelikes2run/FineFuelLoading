# app.py  v11
# CONUS PSA Herbaceous (HER) API — RAP 16-day provisional vs PSA normals CSV
#
# Env vars expected on Render:
#   EE_SERVICE_ACCOUNT      e.g. finefuel@finefuelloading.iam.gserviceaccount.com
#   EE_PRIVATE_KEY_FILE     e.g. /etc/secrets/ee-key.json
#   PSA_NORMALS_CSV         defaults to psa_HER_norm_CONUS_v1.csv
#   PSA_LITE_GEOJSON        defaults to PSA_CONUS_lite.geojson
#   EE_COLLECTION_16D_PROV  defaults to …npp-partitioned-16day-v3-provisional
#   CACHE_REFRESH_HOURS     how often to refresh the cache (default 6)
#   HTTP_TIMEOUT_SEC        defaults to 60
#   CORS_ALLOW_ORIGINS      defaults to "*"
#
# CHANGES v10 → v11:
#   KEY FIX: replaced live ArcGIS geometry fetch with PSA_CONUS_lite.geojson
#   (already in repo, 0.94MB, 98% fewer vertices than full-res ArcGIS data).
#   Loading full-res polygons from ArcGIS at runtime was the primary cause of
#   OOM crashes — each PSA had ~4,600 vertices being held in Python memory
#   while constructing ee.Feature objects. The lite file has ~106 pts/PSA,
#   cutting geometry memory by ~97%.
#   Also added gc.collect() between GACC iterations to release memory promptly.

from __future__ import annotations
import gc, io, json, os, threading, time, uuid, datetime
from typing import List, Optional

import pandas as pd
import requests
import ee
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────
PSA_LITE_GEOJSON     = os.getenv("PSA_LITE_GEOJSON", "PSA_CONUS_lite.geojson")
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
app = FastAPI(title="CONUS PSA Herbaceous API", version="11.0.0")
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

# ── Load lite PSA GeoJSON (done once at startup) ──────────────────────────────
# PSA_CONUS_lite.geojson is pre-simplified (106 pts/PSA vs 4,600 in ArcGIS).
# Using this instead of live ArcGIS fetches cuts geometry memory by ~97%.
def load_lite_geojson(path: str) -> list:
    """Return list of GeoJSON feature dicts from the lite file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lite GeoJSON not found: {path}")
    with open(path, "r") as f:
        gj = json.load(f)
    features = gj.get("features", [])
    # Filter junk rows
    clean = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        psa_code = str(props.get("PSANationalCode", props.get("PSA_NAT_CODE", ""))).strip()
        if feat.get("geometry") and not is_junk_psa(psa_code):
            clean.append(feat)
    print(f"Lite GeoJSON: {len(clean)} valid PSA features loaded from {path}")
    return clean

# ── Build ee.FeatureCollection from lite features for one GACC ───────────────
def get_psa_fc_from_lite(lite_features: list, gacc: Optional[str] = None) -> ee.FeatureCollection:
    """
    Build an ee.FeatureCollection from pre-loaded lite GeoJSON features.
    Optionally filter to a single GACC. Geometries are already simplified
    (~106 pts/PSA) so no further simplify() call is needed.
    """
    ee_feats = []
    for f in lite_features:
        props    = f.get("properties", {}) or {}
        geom     = f.get("geometry")
        psa_code = str(props.get("PSANationalCode", props.get("PSA_NAT_CODE", ""))).strip()
        gacc_id  = str(props.get("GACCUnitID", props.get("GACC", ""))).strip()
        psaname  = props.get("PSANAME", props.get("PSA_NAME", ""))

        if not geom or is_junk_psa(psa_code):
            continue
        if gacc and gacc_id != gacc:
            continue

        psa_key = psa_code.upper()
        ee_feats.append(ee.Feature(ee.Geometry(geom), {
            "PSA_KEY":         psa_key,
            "PSANationalCode": psa_key,
            "PSANAME":         psaname,
            "GACCUnitID":      gacc_id,
        }))
    return ee.FeatureCollection(ee_feats)

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

LITE_FEATURES: list = []
try:
    LITE_FEATURES = load_lite_geojson(PSA_LITE_GEOJSON)
except Exception as e:
    print(f"Lite GeoJSON load failed: {e}")

# ── Core compute (background thread only) ────────────────────────────────────
def _compute_all_conus():
    """
    Processes one GACC at a time using pre-loaded lite geometries.
    Peak memory stays low; gc.collect() between GACCs releases objects promptly.
    Never called on the HTTP request path — no 502 risk.
    """
    print("Cache: starting CONUS computation (lite geojson + per-GACC) ...")
    try:
        if not EE_READY:
            raise RuntimeError("Earth Engine not initialized")
        if NORMALS_DF is None or NORMALS_DF.empty:
            raise RuntimeError("Normals table not loaded")
        if not LITE_FEATURES:
            raise RuntimeError("Lite GeoJSON not loaded")

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
                psa_fc  = get_psa_fc_from_lite(LITE_FEATURES, gacc=gacc)
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

            finally:
                # Force GC between GACCs to release ee.Feature objects promptly
                gc.collect()

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

        print(f"Cache: ready — {len(rows)} PSA rows stored")

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


if EE_READY and NORMALS_DF is not None and LITE_FEATURES:
    t = threading.Thread(target=_cache_refresh_loop, daemon=True)
    t.start()
    print("Cache: background refresh thread started")
else:
    print("Cache: skipping background thread (EE, normals, or lite GeoJSON not ready)")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "CONUS PSA Herbaceous API v11\n\n"
        "Endpoints:\n"
        "  /psa_flags                  All CONUS (served from cache)\n"
        "  /psa_flags?gaccs=USCAONCC  Filter by GACC (comma-separated)\n"
        "  /psa_flags?pretty=1         Human-readable JSON\n"
        "  /health                     Cache status + row count + per-GACC progress\n"
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
        "lite_geojson":     PSA_LITE_GEOJSON,
        "lite_features":    len(LITE_FEATURES),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")))
