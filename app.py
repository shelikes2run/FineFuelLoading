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
import time
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

# ---------------- Logging ----------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ca-psa-her-api")

# ---------------- Config -----------------
PSA_NORMALS_CSV = os.environ.get("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")
EE_COLLECTION_16D_PROV = os.environ.get(
    "EE_COLLECTION_16D_PROV",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional",
)
CACHE_TTL = int(os.environ.get("CACHE_TTL", "900"))
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT_SEC", "60"))

# PSA boundaries (GeoJSON)
PSA_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

# ---------------- Tiny TTL cache ---------
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

# ---------------- Earth Engine init ------
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

# ---------------- FastAPI ----------------
app = FastAPI(title="CA PSA Herbaceous API", version="3.3.2", openapi_url="/openapi.json")


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
    if "features" not in gj or not gj["features"]:
        raise RuntimeError("No PSA features returned.")
    return gj


def ee_latest_her_by_psa(psa_geojson: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute latest HER (=afgNPP+pfgNPP) mean per PSA from RAP 16-day provisional.
    Returns DataFrame: PSANAME, GACCUnitID, HER_latest, afgNPP_latest, pfgNPP_latest
    """
    if not ee_ok:
        raise RuntimeError("Earth Engine is not initialized")

    col = ee.ImageCollection(EE_COLLECTION_16D_PROV)
    latest_img = col.sort("system:time_start", False).first()
    if latest_img is None:
        raise RuntimeError("No images in RAP 16-day collection")

    afg = latest_img.select("afgNPP")
    pfg = latest_img.select("pfgNPP")
    her = afg.add(pfg).rename("HER")
    stack = her.addBands(afg).addBands(pfg)  # HER, afgNPP, pfgNPP

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
            ee_geom = ee.Geometry(geom)
            feats.append(ee.Feature(ee_geom, {"PSANAME": psaname, "GACCUnitID": gacc}))
        except Exception:
            continue

    fc = ee.FeatureCollection(feats)

    reduced = stack.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=30, bestEffort=True
    )
    res = reduced.getInfo()  # dict with "features"
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
    return pd.DataFrame(rows)


def load_normals() -> pd.DataFrame:
    """Load baseline normals CSV."""
    if not os.path.exists(PSA_NORMALS_CSV):
        raise FileNotFoundError(f"Normals CSV not found: {PSA_NORMALS_CSV}")
    df = pd.read_csv(PSA_NORMALS_CSV)
    for col in ["PSANAME", "GACCUnitID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "HER_norm" not in df.columns:
        if {"afgNPP_norm", "pfgNPP_norm"}.issubset(df.columns):
            df["HER_norm"] = df["afgNPP_norm"].fillna(0) + df["pfgNPP_norm"].fillna(0)
        else:
            df["HER_norm"] = 0.0
    return df


def compute_latest_flags_for(gacc_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """PSA polygons -> EE latest means -> merge normals -> flags."""
    # 1) PSA polygons (cache)
    gj = cache.get("psa_geojson")
    if gj is None:
        log.info("Fetching PSA polygons (GeoJSON) from ArcGIS…")
        gj = fetch_psa_geojson()
        cache.set("psa_geojson", gj)

    # 2) Latest EE reduction (cache coarse sig)
    latest_df = cache.get("latest_df")
    if latest_df is None:
        try:
            log.info("Reducing EE latest composite by PSA…")
            latest_df = ee_latest_her_by_psa(gj)
        except Exception as e:
            log.error("EE latest reduction failed: %s", e)
            latest_df = pd.DataFrame(
                columns=[
                    "PSANAME",
                    "GACCUnitID",
                    "HER_latest",
                    "afgNPP_latest",
                    "pfgNPP_latest",
                ]
            )
        cache.set("latest_df", latest_df)

    # 3) Normals
    normals = load_normals()

    # 4) Optional GACC filter (use normals as canonical)
    if gacc_list:
        gacc_norm = [g.strip().upper() for g in gacc_list if g.strip()]
        if "GACCUnitID" in normals.columns:
            normals = normals[
                normals["GACCUnitID"].astype(str).str.upper().isin(gacc_norm)
            ].copy()

    # 5) Merge
    on_cols = []
    if "PSANAME" in normals.columns:
        on_cols.append("PSANAME")
    if "GACCUnitID" in normals.columns:
        on_cols.append("GACCUnitID")
    if not on_cols:
        on_cols = ["PSANAME"]

    merged = normals.merge(latest_df, on=on_cols, how="left")
    log.info("Merged latest RAP (%d rows) with normals (%d rows)", len(latest_df), len(normals))

    # ---- SAFETY: make all numeric, kill NaN/inf before JSON
    for c in ["HER_latest", "afgNPP_latest", "pfgNPP_latest", "HER_norm", "afgNPP_norm", "pfgNPP_norm"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged = merged.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    # Compute flag
    if "HER_latest" not in merged.columns:
        merged["HER_latest"] = 0.0
    if "HER_norm" not in merged.columns:
        merged["HER_norm"] = 0.0
    merged["above_normal"] = merged["HER_latest"] > merged["HER_norm"]

    # Output
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

    rows = (
        merged[existing]
        .sort_values(["GACCUnitID", "PSANAME"], na_position="last")
        .to_dict(orient="records")
    )

    # Final JSON safety: ensure native floats & no NaN/inf
    for r in rows:
        for k, v in list(r.items()):
            # pandas/numpy may leave special floats; sanitize
            try:
                if v is None:
                    r[k] = 0.0
                elif isinstance(v, float):
                    if (v != v) or v in (float("inf"), float("-inf")):
                        r[k] = 0.0
                elif isinstance(v, (int,)):
                    r[k] = float(v)
            except Exception:
                r[k] = 0.0

    return {"count": len(rows), "rows": rows}


# ------------------- Routes -------------------

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
def psa_flags(
    gaccs: Optional[str] = Query(
        None, description="Comma-separated GACCs, e.g. USCAOSCC,USCAONCC"
    )
):
    try:
        gacc_list = [g.strip() for g in gaccs.split(",")] if gaccs else None
        payload = compute_latest_flags_for(gacc_list)
        return JSONResponse(payload)
    except Exception as e:
        log.exception("psa_flags failed")
        raise HTTPException(status_code=500, detail=str(e))

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


# (Render runs with: uvicorn app:app --host 0.0.0.0 --port $PORT)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
