# app.py
# CA PSA Herbaceous (HER) API — uses RAP 16-day provisional (afgNPP/pfgNPP) vs PSA normals CSV
# Render env vars expected:
#   EE_SERVICE_ACCOUNT        (e.g., finefuel@finefuelloading.iam.gserviceaccount.com)
#   EE_PRIVATE_KEY_FILE       (/etc/secrets/ee-key.json)
#   PSA_NORMALS_CSV           (default: psa_HER_norm_CA_v3.csv)
#   EE_COLLECTION_16D_PROV    (default: projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional)
#   CACHE_TTL                 (seconds; default 900)

import os
import json
import math
import time
from typing import List, Optional, Dict, Any

import pandas as pd
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# -------------------------
# Config via environment
# -------------------------
EE_COLLECTION_16D_PROV = os.getenv(
    "EE_COLLECTION_16D_PROV",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional",
)
EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT", "")
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "")  # e.g., /etc/secrets/ee-key.json
PSA_NORMALS_CSV = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "900"))  # 15 minutes
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))  # serialized by default
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")

# -------------------------
# Optional Earth Engine init
# -------------------------
EE_OK = False
try:
    import ee  # type: ignore

    if EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE:
        credentials = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_FILE)
        ee.Initialize(credentials)
        EE_OK = True
    else:
        # Try application default / no-auth init
        ee.Initialize()
        EE_OK = True
except Exception as _ee_err:
    EE_OK = False

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="CA PSA Herbaceous API", version="3.3.1")


# -------------------------
# Simple TTL cache
# -------------------------
class TTLCache:
    def __init__(self, ttl: int = 600, max_entries: int = 64):
        self.ttl = ttl
        self.max_entries = max_entries
        self._store: Dict[str, Any] = {}
        self._time: Dict[str, float] = {}

    def get(self, key: str):
        now = time.time()
        if key in self._store and (now - self._time.get(key, 0)) < self.ttl:
            return self._store[key]
        if key in self._store:
            # expired
            self._store.pop(key, None)
            self._time.pop(key, None)
        return None

    def set(self, key: str, value: Any):
        if len(self._store) >= self.max_entries:
            # drop oldest
            oldest = sorted(self._time.items(), key=lambda kv: kv[1])[0][0]
            self._store.pop(oldest, None)
            self._time.pop(oldest, None)
        self._store[key] = value
        self._time[key] = time.time()


cache = TTLCache(ttl=CACHE_TTL, max_entries=64)

# -------------------------
# Normals CSV (afg/pfg + HER)
# -------------------------
def load_normals_df() -> pd.DataFrame:
    """
    Expected CSV columns:
      PSANAME,GACCUnitID,afgNPP_norm,pfgNPP_norm,HER_norm
    """
    if not os.path.exists(PSA_NORMALS_CSV):
        raise FileNotFoundError(f"Normals CSV not found: {PSA_NORMALS_CSV}")

    df = pd.read_csv(PSA_NORMALS_CSV)
    # Standardize column names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    for c in ["afgNPP_norm", "pfgNPP_norm"]:
        if c not in df.columns:
            # If older file only has HER_norm, split evenly (best-effort)
            df[c] = pd.NA
    if "HER_norm" not in df.columns:
        df["HER_norm"] = df.get("afgNPP_norm", 0).fillna(0) + df.get("pfgNPP_norm", 0).fillna(0)
    return df


NORMALS_DF = load_normals_df()


# -------------------------
# PSA FeatureServer -> ee.FeatureCollection
# Keeps PSANAME/GACCUnitID/PSAID
# -------------------------
PSA_FS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

def fetch_psa_geojson(gaccs: List[str]) -> Dict[str, Any]:
    where = "1=1"
    if gaccs:
        quoted = ",".join([f"'{g}'" for g in gaccs])
        where = f"GACCUnitID IN ({quoted})"
    params = {
        "where": where,
        "outFields": "PSANAME,GACCUnitID,PSAID",
        "f": "geojson",
        "returnGeometry": "true",
        "outSR": 4326,
    }
    r = requests.get(PSA_FS_URL, params=params, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    gj = r.json()
    if "features" not in gj or not gj["features"]:
        return {"type": "FeatureCollection", "features": []}
    return gj


def load_psa_fc(gaccs: List[str]):
    """
    Returns ee.FeatureCollection with PSANAME/GACCUnitID/PSAID properties intact.
    Geometries are simplified a bit to reduce payload.
    """
    if not EE_OK:
        raise RuntimeError("Earth Engine not initialized.")
    gj = fetch_psa_geojson(gaccs)
    fc = ee.FeatureCollection(gj).map(
        lambda f: f.setGeometry(ee.Geometry(f.geometry()).simplify(250))
    )
    # ensure we keep only needed props (but still a Feature)
    fc = fc.select(["PSANAME", "GACCUnitID", "PSAID"], None, False)
    return fc


# -------------------------
# EE helpers
# -------------------------
def ee_latest_16d_image():
    """Latest composite in the RAP 16-day provisional collection."""
    if not EE_OK:
        raise RuntimeError("Earth Engine not initialized.")
    col = ee.ImageCollection(EE_COLLECTION_16D_PROV).sort("system:time_start", False)
    img = col.first()
    return img

def ee_image_date_ymd(img) -> str:
    """Return YYYY-MM-DD for an ee.Image system:time_start."""
    d = ee.Date(img.get("system:time_start"))
    return d.format("YYYY-MM-dd").getInfo()

def add_herbaceous_bands(img):
    """
    Make sure the herbaceous bands exist on the image we reduce.
    Expected band names: 'afgNPP' (annual forb/grass) and 'pfgNPP' (perennial forb/grass).
    Also derive HER = afg + pfg.
    """
    # If bands already exist, just derive HER.
    band_names = ee.List(img.bandNames())
    has_afg = band_names.contains("afgNPP")
    has_pfg = band_names.contains("pfgNPP")

    def _with_her(i):
        afg = i.select("afgNPP")
        pfg = i.select("pfgNPP")
        her = afg.add(pfg).rename("HER")
        return i.addBands(her, overwrite=True)

    return ee.Image(ee.Algorithms.If(has_afg.And(has_pfg), _with_her(img), img))


def ee_mean_by_polygons(img, fc):
    """
    Reduce mean for bands of interest by PSA polygons, preserving input properties.
    """
    target = add_herbaceous_bands(img).select(["afgNPP", "pfgNPP", "HER"])
    reduced = target.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=4,  # safe and avoids unsupported args
    )
    return reduced


# -------------------------
# Comparison & payload
# -------------------------
def compute_latest_flags_for(gacc_list: List[str]) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "count": N, "gaccs": [...],
        "collection": "...",
        "latest_composite": "YYYY-MM-DD",
        "rows": [ {PSANAME, GACCUnitID, PSAID, afgNPP_latest, pfgNPP_latest, HER_latest,
                   afgNPP_norm, pfgNPP_norm, HER_norm, above_normal } ...]
      }
    """
    if not EE_OK:
        raise HTTPException(status_code=500, detail="Earth Engine is not initialized.")

    # Cache key
    key = f"flags::{','.join(sorted(gacc_list)) or 'ALL'}"
    cached = cache.get(key)
    if cached:
        return cached

    # Load PSAs
    psa_fc = load_psa_fc(gacc_list)
    # Short-circuit if nothing returned
    size = psa_fc.size().getInfo()
    if size == 0:
        return {"count": 0, "gaccs": gacc_list, "collection": EE_COLLECTION_16D_PROV, "latest_composite": None, "rows": []}

    # Latest image + date
    img = ee_latest_16d_image()
    latest_date = ee_image_date_ymd(img)

    # Reduce
    reduced_fc = ee_mean_by_polygons(img, psa_fc)

    # Gather rows
    features = reduced_fc.getInfo().get("features", [])
    out_rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        psaname = props.get("PSANAME")
        gacc = props.get("GACCUnitID")
        psaid = props.get("PSAID")

        afg_latest = props.get("afgNPP_mean")
        pfg_latest = props.get("pfgNPP_mean")
        her_latest = props.get("HER_mean")

        # Some EE backends name outputs "<band>_mean"
        if afg_latest is None and props.get("mean") is not None and "afgNPP" in props:
            afg_latest = props.get("mean")

        # Pull normals from CSV (match on PSANAME + GACCUnitID)
        norm_row = NORMALS_DF[
            (NORMALS_DF["PSANAME"] == psaname) & (NORMALS_DF["GACCUnitID"] == gacc)
        ]
        if norm_row.empty:
            afg_norm = None
            pfg_norm = None
            her_norm = None
        else:
            afg_norm = to_float(norm_row.iloc[0].get("afgNPP_norm"))
            pfg_norm = to_float(norm_row.iloc[0].get("pfgNPP_norm"))
            her_norm = to_float(norm_row.iloc[0].get("HER_norm"))

        # If HER not computed or NaN, recompute
        if her_latest is None and afg_latest is not None and pfg_latest is not None:
            her_latest = safe_add(afg_latest, pfg_latest)

        above = 0
        if her_norm is not None and her_latest is not None:
            above = 1 if float(her_latest) > float(her_norm) else 0

        out_rows.append({
            "PSANAME": psaname,
            "GACCUnitID": gacc,
            "PSAID": psaid,
            "afgNPP_latest": to_float(afg_latest),
            "pfgNPP_latest": to_float(pfg_latest),
            "HER_latest": to_float(her_latest),
            "afgNPP_norm": to_float(afg_norm),
            "pfgNPP_norm": to_float(pfg_norm),
            "HER_norm": to_float(her_norm),
            "above_normal": int(above),
        })

    payload = {
        "count": len(out_rows),
        "gaccs": gacc_list,
        "collection": EE_COLLECTION_16D_PROV,
        "latest_composite": latest_date,
        "rows": out_rows,
    }
    cache.set(key, payload)
    return payload


def to_float(x):
    """Convert to JSON-safe float (None if NaN/inf)."""
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def safe_add(a, b):
    try:
        if a is None or b is None:
            return None
        va, vb = float(a), float(b)
        if any(map(math.isnan, [va, vb])) or any(map(math.isinf, [va, vb])):
            return None
        return va + vb
    except Exception:
        return None


# -------------------------
# Schemas
# -------------------------
class FlagsResponse(BaseModel):
    count: int
    gaccs: List[str]
    collection: Optional[str]
    latest_composite: Optional[str]
    rows: List[Dict[str, Any]]


# -------------------------
# Routes
# -------------------------
@app.get("/", tags=["root"])
def root():
    return {
        "name": "CA PSA Herbaceous API",
        "version": app.version,
        "endpoints": [
            "/health",
            "/psa_flags?gaccs=USCAOSCC,USCAONCC&pretty=1",
        ],
    }


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "ee_initialized": EE_OK,
        "normals_csv": PSA_NORMALS_CSV,
        "cache_ttl_sec": CACHE_TTL,
    }


@app.get("/psa_flags", response_model=FlagsResponse, tags=["psas"])
def psa_flags(
    gaccs: Optional[str] = Query(default=None, description="Comma-separated list of GACCUnitID values (e.g., USCAOSCC,USCAONCC). If omitted, all PSAs."),
    pretty: Optional[int] = Query(default=0, ge=0, le=1, description="Pretty-print JSON if =1."),
):
    if not EE_OK:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized on server.")

    gacc_list = []
    if gaccs:
        gacc_list = [g.strip() for g in gaccs.split(",") if g.strip()]

    try:
        payload = compute_latest_flags_for(gacc_list)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"PSA FeatureServer error: {e}")
    except ee.ee_exception.EEException as e:  # type: ignore
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # pretty=1 support
    if pretty == 1:
        return JSONResponse(
            content=json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False)),
            media_type="application/json",
            headers={"X-Pretty-Print": "1"},
        )
    return payload


# -------------------------
# Uvicorn entry (Render)
# -------------------------
def main():
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()
