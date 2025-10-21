import os
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# CONFIG
# ============================================================
NIFC_PSA_URL = os.getenv(
    "NIFC_PSA_URL",
    "https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/"
    "Predictive_Service_Area_PSA_Boundaries_Public/FeatureServer/0/query",
)
RAP_URL = os.getenv(
    "RAP_URL",
    "https://us-central1-rap-data-365417.cloudfunctions.net/production16dayV3",
)

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "120"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "1.7"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "21600"))  # 6 hours

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(title="CA PSA Herbaceous API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SIMPLE IN-MEMORY CACHE
# ============================================================
_cache: Dict[str, Tuple[float, Any]] = {}  # key -> (expires_at, value)

def cache_get(key: str):
    now = time.time()
    hit = _cache.get(key)
    if not hit:
        return None
    exp, val = hit
    if now > exp:
        _cache.pop(key, None)
        return None
    return val

def cache_set(key: str, val: Any, ttl: int = CACHE_TTL_SEC):
    _cache[key] = (time.time() + ttl, val)

# ============================================================
# HTTP HELPERS
# ============================================================
async def _http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            last_exc = e
            await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
    raise HTTPException(status_code=502, detail=f"GET failed for {url}: {last_exc}")

async def _http_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            last_exc = e
            await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
    raise HTTPException(status_code=502, detail=f"POST failed for {url}: {last_exc}")

# ============================================================
# HELPERS
# ============================================================
def _arcgis_query_params(where: str, f: str = "json", out_fields: str = "*", return_geometry: bool = True):
    """Build standard ArcGIS REST query parameters."""
    return {
        "where": where,                # we will use "1=1" and filter client-side
        "outFields": out_fields,       # safest on AO layers
        "f": f,                        # Esri JSON (more reliable than geojson output)
        "returnGeometry": "true" if return_geometry else "false",
        "outSR": 4326,
    }

def _esri_polygon_to_geojson(esri_geom: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Esri polygon rings to GeoJSON Polygon."""
    rings = esri_geom.get("rings") or []
    if not rings:
        raise ValueError("Empty Esri polygon.")
    def _close(coords):
        return coords if coords[0] == coords[-1] else coords + [coords[0]]
    shell = _close([[x, y] for x, y in rings[0]])
    holes = []
    if len(rings) > 1:
        holes = [_close([[x, y] for x, y in r]) for r in rings[1:]]
    return {"type": "Polygon", "coordinates": [shell] + holes}

def _last_n(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    return rows[-n:] if n and n < len(rows) else rows

async def _rap_for_polygon(geojson_polygon: Dict[str, Any], year: Optional[int]) -> Dict[str, Any]:
    payload = {"aoi": geojson_polygon}
    if year:
        payload["year"] = int(year)
    return await _http_post_json(RAP_URL, payload)

async def fetch_ca_psas(gacc: Optional[str] = None) -> Dict[str, Any]:
    """
    Get PSA features as Esri JSON (no server-side filter), then filter to CA + desired GACC(s) in code.
    This avoids 400s some services return on certain where clauses or field-name case issues.
    """
    params = _arcgis_query_params(
        where="1=1",      # no filter here
        f="json",
        out_fields="*",
        return_geometry=True
    )
    data = await _http_get_json(NIFC_PSA_URL, params)

    # Surface ArcGIS error objects if present
    if isinstance(data, dict) and "error" in data:
        msg = data["error"].get("message", "ArcGIS error")
        details = data["error"].get("details", [])
        raise HTTPException(status_code=502, detail=f"NIFC PSA error: {msg} {details}")

    if "features" not in data:
        raise HTTPException(status_code=502, detail="Unexpected PSA response (no 'features')")

    # Client-side filter
    wanted_gaccs = None
    if gacc:
        wanted_gaccs = {x.strip().upper() for x in gacc.split(",")}

    filtered = []
    for f in data["features"]:
        attrs = f.get("attributes", {}) or {}
        state = (attrs.get("STATE") or attrs.get("State") or attrs.get("state"))
        gacc_val = (attrs.get("GACC") or attrs.get("gacc"))
        if state == "CA" and (wanted_gaccs is None or (gacc_val and gacc_val.upper() in wanted_gaccs)):
            filtered.append(f)

    return {"features": filtered}

# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
async def health():
    return {"ok": True, "ts": int(time.time())}

@app.get("/psas")
async def psas(gacc: Optional[str] = Query(None, description="Comma-delimited GACCs (e.g., 'ONCC,OSCC'). Default: both.")):
    """Return filtered CA PSAs (Esri JSON)."""
    return await fetch_ca_psas(gacc=gacc)

@app.get("/ca_psa_herbaceous")
async def ca_psa_herbaceous(
    year: Optional[int] = Query(None, description="Optional RAP year filter."),
    intervals: int = Query(3, ge=1, le=12, description="Number of most-recent 16-day intervals to include."),
    gacc: Optional[str] = Query(None, description="Comma-delimited GACCs (default ONCC,OSCC)."),
    include_series: bool = Query(True, description="If false, only latest values are returned."),
):
    """Return RAP herbaceous production (AFG/PFG/HER, lbs/ac) by CA PSA as GeoJSON."""
    cache_key = f"psa:{gacc}|year:{year}|int:{intervals}|series:{include_series}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    psa_fc = await fetch_ca_psas(gacc=gacc)
    feats = psa_fc["features"]
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(f: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        attrs = f.get("attributes", {}) or {}
        esri_geom = f.get("geometry", {}) or {}
        if "rings" not in esri_geom:
            return None
        try:
            geojson_poly = _esri_polygon_to_geojson(esri_geom)
        except Exception:
            return None

        async with sem:
            rap_json = await _rap_for_polygon(geojson_poly, year)
        series = _last_n(rap_json.get("production16day", []), intervals)

        rec = {
            "PSA_ID":   attrs.get("PSA_ID") or attrs.get("PSAID"),
            "PSA_NAME": attrs.get("PSA_NAME") or attrs.get("PSANAME"),
            "GACC":     attrs.get("GACC"),
            "STATE":    attrs.get("STATE"),
        }

        latest = series[-1] if series else None
        if latest:
            rec["AFG_latest"] = latest.get("AFG")
            rec["PFG_latest"] = latest.get("PFG")
            rec["HER_latest"] = latest.get("HER")
            rec["IntervalDate_latest"] = latest.get("date")

        if include_series:
            for row in series:
                d = row.get("date")
                if d:
                    rec[f"AFG_lbsac_{d}"] = row.get("AFG")
                    rec[f"PFG_lbsac_{d}"] = row.get("PFG")
                    rec[f"HER_lbsac_{d}"] = row.get("HER")

        return {"type": "Feature", "geometry": geojson_poly, "properties": rec}

    results = await asyncio.gather(*(worker(f) for f in feats))
    out_features = [r for r in results if r]
    out = {"type": "FeatureCollection", "features": out_features}
    cache_set(cache_key, out)
    return out

# ============================================================
# SOUTH OPS PL POINTS ENDPOINT
# ============================================================
def _pl_points_from_count(affected_count: int) -> int:
    """South Ops scale mapping."""
    if affected_count <= 1:
        return 2
    if affected_count <= 4:
        return 4
    if affected_count <= 7:
        return 6
    if affected_count <= 10:
        return 8
    return 10

@app.get("/southops_pl_points")
async def southops_pl_points(
    threshold_lbsac: float,
    intervals: int = Query(1, ge=1, le=12, description="How many intervals to fetch (use latest interval for scoring)."),
    metric: str = Query("HER", pattern="^(HER|AFG|PFG)$", description="Metric to test (HER, AFG, or PFG)."),
    gacc: str = Query("OSCC", description="Which GACC(s) to evaluate; default OSCC (South Ops)."),
):
    """
    Count PSAs in chosen GACC(s) with <metric>_latest >= threshold_lbsac,
    then return PL points using South Ops scale.
    """
    fc = await ca_psa_herbaceous(year=None, intervals=intervals, gacc=gacc, include_series=False)
    feats = fc.get("features", [])
    total_psas = len(feats)
    affected = []
    latest_date = None

    for f in feats:
        props = f["properties"]
        latest_date = latest_date or props.get("IntervalDate_latest")
        val = props.get(f"{metric}_latest")
        if val is not None and float(val) >= float(threshold_lbsac):
            affected.append({
                "PSA_ID": props.get("PSA_ID"),
                "PSA_NAME": props.get("PSA_NAME"),
                "GACC": props.get("GACC"),
                f"{metric}_latest": val,
            })

    affected_count = len(affected)
    points = _pl_points_from_count(affected_count)

    return {
        "gacc": gacc,
        "metric": metric,
        "threshold_lbsac": threshold_lbsac,
        "total_psas": total_psas,
        "affected_count": affected_count,
        "points": points,
        "scale": {"0-1": 2, "2-4": 4, "5-7": 6, "8-10": 8, "11+": 10},
        "as_of_interval_date": latest_date,
        "affected_psas": affected,
    }
