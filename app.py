# app.py  v14
# California PSA Herbaceous (HER) API — RAP public API vs PSA normals CSV
#
# NO GOOGLE EARTH ENGINE REQUIRED — uses RAP's free public API directly.
#
# Env vars expected on Render:
#   PSA_NORMALS_CSV         defaults to psa_HER_norm_CA_v2.csv
#   PSA_LITE_GEOJSON        defaults to PSA_CONUS_lite.geojson
#   CACHE_REFRESH_HOURS     how often to refresh the cache (default 6)
#   RAP_REQUEST_DELAY_SEC   delay between RAP API calls (default 0.5)
#   HTTP_TIMEOUT_SEC        defaults to 60
#   CORS_ALLOW_ORIGINS      defaults to "*"
#
# CHANGES v13 → v14:
#   Removed Google Earth Engine entirely. Now calls the RAP public API:
#   POST https://us-central1-rap-data-365417.cloudfunctions.net/production16dayV3
#   This endpoint accepts a GeoJSON polygon and returns 16-day AFG/PFG/HER
#   production values (lbs/acre). No credentials or account required.
#   Normals CSV must be regenerated using the companion notebook
#   (generate_normals_RAP_API.ipynb) which uses the same API and units.

from __future__ import annotations
import gc, json, os, threading, time, datetime
from typing import List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────
RAP_16DAY_URL    = "https://us-central1-rap-data-365417.cloudfunctions.net/production16dayV3"
PSA_LITE_GEOJSON = os.getenv("PSA_LITE_GEOJSON", "PSA_CONUS_lite.geojson")
NORMALS_CSV      = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CA_v2.csv")
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))
CACHE_REFRESH_HOURS  = float(os.getenv("CACHE_REFRESH_HOURS", "6"))
RAP_REQUEST_DELAY    = float(os.getenv("RAP_REQUEST_DELAY_SEC", "0.5"))

GACC_CODES = [
    "USCAONCC",  # California North
    "USCAOSCC",  # California South
]

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="California PSA Herbaceous API", version="14.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ── In-memory cache ───────────────────────────────────────────────────────────
_CACHE: dict = {
    "rows": [], "latest_composite": None, "computed_at": None,
    "status": "starting", "error": None, "gacc_progress": {},
}
_CACHE_LOCK = threading.Lock()

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
    required = ["PSANationalCode","PSANAME","GACCUnitID","afgNPP_norm","pfgNPP_norm","HER_norm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Normals CSV missing column: {col}")
    df = df[df["PSANationalCode"].notna()]
    df = df[~df["PSANationalCode"].astype(str).apply(is_junk_psa)]
    df["PSA_KEY"] = df["PSANationalCode"].astype(str).str.upper().str.strip()
    return df[["PSA_KEY","PSANationalCode","PSANAME","GACCUnitID",
               "afgNPP_norm","pfgNPP_norm","HER_norm"]]

# ── Load lite PSA GeoJSON (once at startup) ───────────────────────────────────
def load_lite_geojson(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lite GeoJSON not found: {path}")
    with open(path, "r") as f:
        gj = json.load(f)
    clean = []
    for feat in gj.get("features", []):
        props    = feat.get("properties", {}) or {}
        psa_code = str(props.get("PSANationalCode", props.get("PSA_NAT_CODE", ""))).strip()
        if feat.get("geometry") and not is_junk_psa(psa_code):
            clean.append(feat)
    print(f"Lite GeoJSON: {len(clean)} valid PSA features loaded from {path}")
    return clean

# ── RAP public API call ───────────────────────────────────────────────────────
def fetch_rap_16day(geometry: dict, year: Optional[int] = None) -> list:
    """
    POST a GeoJSON polygon to the RAP production16dayV3 endpoint.
    Returns a list of rows: [date, year, doy, AFG, PFG, HER]
    The first row is the header. Subsequent rows are data.
    No credentials required — fully public API.
    """
    payload = {
        "type": "Feature",
        "geometry": geometry,
        "properties": {"mask": True, "year": year},
    }
    r = requests.post(RAP_16DAY_URL, json=payload, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    rows = data.get("properties", {}).get("production16day", [])
    return rows  # first row is header: ["date","year","doy","AFG","PFG","HER"]


def parse_latest_rap_row(rows: list) -> dict:
    """
    Extract the most recent non-zero 16-day production row.
    Returns dict with keys: date, afgNPP_latest, pfgNPP_latest, HER_latest
    """
    if len(rows) < 2:
        return {}
    # rows[0] is header, rows[1:] are data newest-last
    # Find the last row with non-null HER value
    header = rows[0]
    try:
        date_idx = header.index("date")
        afg_idx  = header.index("AFG")
        pfg_idx  = header.index("PFG")
        her_idx  = header.index("HER")
    except ValueError:
        return {}

    latest = None
    for row in reversed(rows[1:]):
        her_val = safe_num(row[her_idx])
        if her_val is not None:
            latest = row
            break

    if not latest:
        return {}

    return {
        "date":           latest[date_idx],
        "afgNPP_latest":  safe_num(latest[afg_idx]),
        "pfgNPP_latest":  safe_num(latest[pfg_idx]),
        "HER_latest":     safe_num(latest[her_idx]),
    }

# ── Startup data load ─────────────────────────────────────────────────────────
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
    For each PSA polygon, calls the RAP production16dayV3 API to get the
    latest 16-day AFG/PFG/HER values, then compares against normals.
    25 CA PSA API calls, ~15 sec with 0.5s delay. No GEE required.
    """
    print("Cache: starting CA PSA computation via RAP API ...")
    try:
        if NORMALS_DF is None or NORMALS_DF.empty:
            raise RuntimeError("Normals table not loaded")
        if not LITE_FEATURES:
            raise RuntimeError("Lite GeoJSON not loaded")

        current_year = datetime.datetime.utcnow().year
        all_latest_rows = []
        gacc_progress   = {}
        latest_date_seen = None

        for gacc in GACC_CODES:
            print(f"Cache: processing {gacc} ...")
            gacc_features = [
                f for f in LITE_FEATURES
                if str((f.get("properties") or {}).get("GACCUnitID", "")).strip() == gacc
            ]
            if not gacc_features:
                gacc_progress[gacc] = {"status": "skipped", "psas": 0}
                continue

            gacc_rows    = []
            gacc_errors  = []

            for feat in gacc_features:
                props    = feat.get("properties", {}) or {}
                psa_code = str(props.get("PSANationalCode", props.get("PSA_NAT_CODE",""))).strip().upper()
                psaname  = props.get("PSANAME", props.get("PSA_NAME",""))
                geom     = feat.get("geometry")

                if not geom or is_junk_psa(psa_code):
                    continue
                try:
                    rows   = fetch_rap_16day(geom, year=current_year)
                    parsed = parse_latest_rap_row(rows)
                    if parsed:
                        if latest_date_seen is None or parsed["date"] > latest_date_seen:
                            latest_date_seen = parsed["date"]
                        gacc_rows.append({
                            "PSA_KEY":         psa_code,
                            "PSANationalCode": psa_code,
                            "PSANAME":         psaname,
                            "GACCUnitID":      gacc,
                            "afgNPP_latest":   parsed["afgNPP_latest"],
                            "pfgNPP_latest":   parsed["pfgNPP_latest"],
                            "HER_latest":      parsed["HER_latest"],
                        })
                    else:
                        print(f"  {psa_code}: no valid production data returned")
                except Exception as psa_exc:
                    print(f"  {psa_code} failed: {psa_exc}")
                    gacc_errors.append(f"{psa_code}: {psa_exc}")

                time.sleep(RAP_REQUEST_DELAY)

            all_latest_rows.extend(gacc_rows)
            if gacc_errors:
                gacc_progress[gacc] = {
                    "status": "partial" if gacc_rows else "error",
                    "psas": len(gacc_rows), "errors": gacc_errors[:3],
                }
            else:
                gacc_progress[gacc] = {"status": "ok", "psas": len(gacc_rows)}

            gc.collect()

        latest_df = pd.DataFrame(all_latest_rows)
        if latest_df.empty:
            raise RuntimeError("No PSA results returned from RAP API")

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
            _CACHE["latest_composite"] = latest_date_seen
            _CACHE["computed_at"]      = datetime.datetime.utcnow().isoformat() + "Z"
            _CACHE["status"]           = "ready"
            _CACHE["error"]            = None
            _CACHE["gacc_progress"]    = gacc_progress

        print(f"Cache: ready — {len(rows)} PSA rows stored, latest composite = {latest_date_seen}")

    except Exception as exc:
        print(f"Cache: compute failed — {exc}")
        with _CACHE_LOCK:
            _CACHE["status"] = "error"
            _CACHE["error"]  = str(exc)


def _cache_refresh_loop():
    while True:
        _compute_all_conus()
        print(f"Cache: next refresh in {CACHE_REFRESH_HOURS}h")
        time.sleep(CACHE_REFRESH_HOURS * 3600)


if NORMALS_DF is not None and LITE_FEATURES:
    t = threading.Thread(target=_cache_refresh_loop, daemon=True)
    t.start()
    print("Cache: background refresh thread started")
else:
    print("Cache: skipping background thread (normals or lite GeoJSON not ready)")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "California PSA Herbaceous API v14 (RAP API — no GEE)\n\n"
        "Endpoints:\n"
        "  /psa_flags                  All CA PSAs (served from cache)\n"
        "  /psa_flags?gaccs=USCAONCC  Filter by GACC (comma-separated)\n"
        "  /psa_flags?gaccs=USCAOSCC  California South only\n"
        "  /psa_flags?pretty=1         Human-readable JSON\n"
        "  /health                     Cache status + per-GACC progress\n"
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
    return {
        "status":            "ok",
        "normals_csv":       NORMALS_CSV,
        "normals_rows":      len(NORMALS_DF) if NORMALS_DF is not None else 0,
        "lite_geojson":      PSA_LITE_GEOJSON,
        "lite_features":     len(LITE_FEATURES),
        "data_source":       "RAP public API (no GEE)",
        "cache_status":      cache_status,
        "cache_psa_rows":    cache_rows,
        "cache_computed_at": cache_computed,
        "latest_composite":  cache_composite,
        "cache_error":       cache_error,
        "gacc_progress":     gacc_progress,
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
        error    = _CACHE["error"]

    if status == "starting":
        raise HTTPException(status_code=503,
            detail="Cache is still computing — try again in a few minutes.")

    # Serve stale data if available, with a warning
    if status == "error" and not rows:
        raise HTTPException(status_code=500,
            detail=f"Cache compute failed and no data available: {error}")

    gacc_list = parse_gaccs_param(gaccs)
    if gacc_list:
        rows = [r for r in rows if r.get("GACCUnitID") in gacc_list]

    result_gaccs = sorted(set(r["GACCUnitID"] for r in rows if r.get("GACCUnitID")))
    payload = {
        "count":            len(rows),
        "gaccs":            gacc_list if gacc_list else result_gaccs,
        "data_source":      "RAP public API",
        "latest_composite": latest,
        "cache_computed_at":computed,
        "rows":             rows,
    }

    if status == "error":
        payload["cache_warning"] = (
            f"Cache refresh failed — serving last known data from {computed}. "
            f"Error: {error}"
        )

    if bool(pretty):
        return PlainTextResponse(json.dumps(payload, indent=2), media_type="application/json")
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")))
