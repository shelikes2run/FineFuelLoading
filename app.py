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
import time
import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

# ---- Logging ---------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ca-psa-her-api")

# ---- Config ----------------------------------------------------------------
PSA_NORMALS_CSV = os.environ.get("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")
EE_COLLECTION_16D_PROV = os.environ.get(
    "EE_COLLECTION_16D_PROV",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional",
)
CACHE_TTL = int(os.environ.get("CACHE_TTL", "900"))
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT_SEC", "60"))

# PSA boundaries (public) – GeoJSON output
PSA_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

# ---- Tiny TTL cache --------------------------------------------------------
class TTLCache:
    def __init__(self, ttl_s: int = 900):
        self.ttl = ttl_s
        self._data: Dict[str, Any] = {}
        self._ts: Dict[str, float] = {}

    def get(self, key: str):
        now = time.time()
        if key in self._data and now - self._ts.get(key, 0) < self.ttl:
            return self._data[key]
        return None

    def set(self, key: str, val: Any):
        self._data[key] = val
        self._ts[key] = time.time()

cache = TTLCache(CACHE_TTL)

# ---- Earth Engine init -----------------------------------------------------
EE_SERVICE_ACCOUNT = os.environ.get("EE_SERVICE_ACCOUNT")
EE_PRIVATE_KEY_FILE = os.environ.get("EE_PRIVATE_KEY_FILE")

ee_ok = False
try:
    import ee  # type: ignore

    if not EE_SERVICE_ACCOUNT or not EE_PRIVATE_KEY_FILE:
        raise RuntimeError("EE service account or key path missing")

    credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_FILE)
    ee.Initialize(credentials)
    ee_ok = True
    log.info("Earth Engine initialized ✅ as %s", EE_SERVICE_ACCOUNT)
except Exception as e:
    log.error("Earth Engine init FAILED ❌ : %s", e)
    ee_ok = False

# ---- FastAPI ---------------------------------------------------------------
app = FastAPI(title="CA PSA Herbaceous API", version="3.3.1", openapi_url="/openapi.json")


def fetch_psa_geojson(limit: int = 10000) -> Dict[str, Any]:
    """Pull PSA polygons (GeoJSON) with essential properties."""
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": 4326,
        "resultRecordCount": limit,
    }
    r = requests.get(PSA_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    gj = r.json()
    # sanity
    if "features" not in gj or not gj["features"]:
        raise RuntimeError("No PSA features returned.")
    return gj


def ee_latest_her_by_psa(psa_geojson: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute latest HER (= afgNPP + pfgNPP) mean per PSA using RAP 16-day provisional.
    Returns DataFrame with columns: PSANAME, GACCUnitID, HER_latest, afgNPP_latest, pfgNPP_latest
    """
    if not ee_ok:
        raise RuntimeError("Earth Engine is not initialized")

    # Collection & latest image
    col = ee.ImageCollection(EE_COLLECTION_16D_PROV)
    latest_img = col.sort("system:time_start", False).first()
    if latest_img is None:
        raise RuntimeError("No images in RAP 16-day collection")

    # Bands: afgNPP, pfgNPP -> HER
    afg = latest_img.select("afgNPP")
    pfg = latest_img.select("pfgNPP")
    her = afg.add(pfg).rename("HER")

    stack = her.addBands(afg).addBands(pfg)  # HER, afgNPP, pfgNPP

    # Convert GeoJSON features to ee.FeatureCollection (keep PSANAME, GACCUnitID)
    feats = []
    for f in psa_geojson.get("features", []):
        props = f.get("properties", {}) or {}
        psaname = (
            props.get("PSANAME")
            or props.get("PSA_NAME")
            or props.get("PSA NAME_TEXT")
            or "Unknown PSA"
        )
        gacc = props.get("GACCUnitID") or props.get("GACC") or props.get("gacc") or ""
        geom = f.get("geometry")

        if not geom or not geom.get("coordinates"):
            continue

        try:
            ee_geom = ee.Geometry(geom)  # GeoJSON polygon/rings are accepted by EE
            feats.append(ee.Feature(ee_geom, {"PSANAME": psaname, "GACCUnitID": gacc}))
        except Exception:
            # Skip malformed geometries
            continue

    fc = ee.FeatureCollection(feats)

    # ReduceRegions: mean per PSA
    # scale 30m (Landsat-based), bestEffort True to avoid memory overflows on large polys
    reduced = stack.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=30, bestEffort=True
    )

    # Pull results to client
    res = reduced.getInfo()  # list of dicts
    rows = []
    for it in res.get("features", []):
        p = it.get("properties", {}) or {}
        rows.append(
            {
                "PSANAME": p.get("PSANAME"),
                "GACCUnitID": p.get("GACCUnitID"),
                "HER_latest": p.get("HER_mean"),
                "afgNPP_latest": p.get("afgNPP_mean"),
                "pfgNPP_latest": p.get("pfgNPP_mean"),
            }
        )

    df = pd.DataFrame(rows)
    return df


def load_normals() -> pd.DataFrame:
    """Load baseline normals CSV (PSANAME,GACCUnitID,afgNPP_norm,pfgNPP_norm,HER_norm)."""
    if not os.path.exists(PSA_NORMALS_CSV):
        raise FileNotFoundError(f"Normals CSV not found: {PSA_NORMALS_CSV}")
    df = pd.read_csv(PSA_NORMALS_CSV)
    # Normalize key columns
    for col in ["PSANAME", "GACCUnitID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Ensure HER_norm exists (if not, compute)
    if "HER_norm" not in df.columns:
        if {"afgNPP_norm", "pfgNPP_norm"}.issubset(df.columns):
            df["HER_norm"] = df["afgNPP_norm"].fillna(0) + df["pfgNPP_norm"].fillna(0)
        else:
            df["HER_norm"] = 0.0
    return df


def compute_latest_flags_for(gacc_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Pull PSA polygons -> EE latest HER/afg/pfg means -> merge with normals -> flags.
    """
    # 1) PSA polygons (cached)
    gj = cache.get("psa_geojson")
    if gj is None:
        log.info("Fetching PSA polygons (GeoJSON) from ArcGIS…")
        gj = fetch_psa_geojson()
        cache.set("psa_geojson", gj)

    # 2) Latest HER by PSA via EE (cached per latest date signature)
    # Build a simple signature based on collection's latest date
    latest_sig = cache.get("latest_sig")
    latest_df = cache.get("latest_df")

    try:
        if latest_sig is None or latest_df is None:
            # compute now
            log.info("Reducing EE latest composite by PSA…")
            latest_df = ee_latest_her_by_psa(gj)
            # If EE returned nothing (rare), safeguard
            if latest_df is None or latest_df.empty:
                log.warning("EE returned 0 PSA rows; proceeding with empty latest_df.")
                latest_df = pd.DataFrame(
                    columns=["PSANAME", "GACCUnitID", "HER_latest", "afgNPP_latest", "pfgNPP_latest"]
                )
            latest_sig = f"ts:{int(time.time())}"  # coarse signature
            cache.set("latest_sig", latest_sig)
            cache.set("latest_df", latest_df)
    except Exception as e:
        log.error("EE latest reduction failed: %s", e)
        # still proceed with empty latest to avoid 500s
        latest_df = pd.DataFrame(
            columns=["PSANAME", "GACCUnitID", "HER_latest", "afgNPP_latest", "pfgNPP_latest"]
        )

    # 3) Load normals
    normals = load_normals()

    # 4) Optional filter by GACC list (USCAOSCC, USCAONCC, etc.)
    if gacc_list:
        gacc_list_norm = [g.strip().upper() for g in gacc_list]
        # Filter normals first (the canonical list)
        if "GACCUnitID" in normals.columns:
            normals = normals[normals["GACCUnitID"].astype(str).str.upper().isin(gacc_list_norm)].copy()

    # 5) Merge normals + latest
    #    Use PSANAME + GACCUnitID as join keys where possible
    on_cols = []
    if "PSANAME" in normals.columns:
        on_cols.append("PSANAME")
    if "GACCUnitID" in normals.columns:
        on_cols.append("GACCUnitID")

    if not on_cols:
        # last resort join by PSANAME
        on_cols = ["PSANAME"]

    merged = normals.merge(latest_df, on=on_cols, how="left")
    log.info("Merging latest RAP (%d rows) with normals (%d rows)", len(latest_df), len(normals))

    # ---- SAFETY: avoid .fillna(value=HER_latest) ValueError
    # Fill numeric latest columns with 0 instead of crashing
    for c in ["HER_latest", "afgNPP_latest", "pfgNPP_latest"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    # Ensure HER_norm exists and numeric
    if "HER_norm" not in merged.columns:
        merged["HER_norm"] = (
            pd.to_numeric(merged.get("afgNPP_norm", 0), errors="coerce").fillna(0.0)
            + pd.to_numeric(merged.get("pfgNPP_norm", 0), errors="coerce").fillna(0.0)
        )
    else:
        merged["HER_norm"] = pd.to_numeric(merged["HER_norm"], errors="coerce").fillna(0.0)

    # Compute flag
    merged["above_normal"] = merged["HER_latest"] > merged["HER_norm"]

    # Minimal payload
    out_cols = [
        "PSANAME",
        "GACCUnitID",
        "afgNPP_latest",
        "pfgNPP_latest",
        "HER_latest",
        "afgNPP_norm",
        "pfgNPP_norm",
        "HER_norm",
        "above_normal",
    ]
    existing = [c for c in out_cols if c in merged.columns]
    rows = merged[existing].sort_values(["GACCUnitID", "PSANAME"], na_position="last").to_dict(orient="records")

    return {
        "count": len(rows),
        "rows": rows,
    }

# --------------------- Routes ----------------------------------------------

@app.get("/")
def root():
    return {"service": "CA PSA Herbaceous API", "version": app.version}

@app.get("/health")
def health():
    ok = ee_ok
    return {
        "status": "ok" if ok else "error",
        "ee_initialized": ok,
        "normals_csv": PSA_NORMALS_CSV,
        "cache_ttl_sec": CACHE_TTL,
    }

@app.get("/psa_flags")
def psa_flags(gaccs: Optional[str] = Query(None, description="Comma-separated GACCs, e.g. USCAOSCC,USCAONCC")):
    try:
        gacc_list = None
        if gaccs:
            gacc_list = [g.strip() for g in gaccs.split(",") if g.strip()]
        payload = compute_latest_flags_for(gacc_list)
        return JSONResponse(payload)
    except Exception as e:
        log.exception("psa_flags failed")
        raise HTTPException(status_code=500, detail=str(e))

# Convenience endpoint: just the latest composite date (best-effort)
@app.get("/latest_info")
def latest_info():
    if not ee_ok:
        raise HTTPException(status_code=503, detail="Earth Engine not initialized")
    try:
        col = ee.ImageCollection(EE_COLLECTION_16D_PROV)
        latest = col.sort("system:time_start", False).first()
        if latest is None:
            return {"latest": None}
        ts = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        return {"latest": ts, "collection": EE_COLLECTION_16D_PROV}
    except Exception as e:
        return {"latest": None, "error": str(e), "collection": EE_COLLECTION_16D_PROV}
