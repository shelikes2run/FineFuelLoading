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
NIFC_PSA_GEOJSON = os.getenv(
    "NIFC_PSA_GEOJSON",
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

RAP_URL = os.getenv(
    "RAP_URL",
    "https://us-central1-rap-data-365417.cloudfunctions.net/production16dayV3",
)

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "90"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "1.7"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "3600"))
MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "8"))

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(title="CA PSA Herbaceous API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CACHE
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
    if len(_cache) >= MAX_CACHE_ENTRIES:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)
    _cache[key] = (time.time() + ttl, val)

# ============================================================
# GLOBAL HTTP CLIENT
# ============================================================
_client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def _startup():
    global _client
    _client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC)

@app.on_event("shutdown")
async def _shutdown():
    global _client
    if _client:
        await _client.aclose()

# ============================================================
# HELPERS
# ============================================================
def _thin_polygon(geom: Dict[str, Any], max_points: int = 3000) -> Dict[str, Any]:
    """Reduce vertex count to avoid huge AOIs."""
    if not geom or "type" not in geom:
        return geom

    def _thin_ring(ring: List[List[float]], max_pts: int) -> List[List[float]]:
        if len(ring) <= max_pts:
            return ring
        step = max(1, len(ring) // max_pts)
        out = ring[::step]
        if out[0] != out[-1]:
            out.append(out[0])
        return out

    if geom["type"] == "Polygon":
        coords = geom.get("coordinates", [])
        if not coords:
            return geom
        shell = _thin_ring(coords[0], max_points)
        holes = [_thin_ring(r, max_points // 5) for r in coords[1:]]
        return {"type": "Polygon", "coordinates": [shell] + holes}

    if geom["type"] == "MultiPolygon":
        polys = []
        for poly in geom.get("coordinates", []):
            if not poly:
                continue
            shell = _thin_ring(poly[0], max_points)
            holes = [_thin_ring(r, max_points // 5) for r in poly[1:]]
            polys.append([shell] + holes)
        if not polys:
            return {"type": "Polygon", "coordinates": []}
        return {"type": "Polygon", "coordinates": polys[0]}

    return geom

async def _http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    global _client
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = await _client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
    raise HTTPException(status_code=502, detail=f"GET failed for {url}: {last_exc}")

async def _http_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    global _client
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = await _client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
    raise HTTPException(status_code=502, detail=f"POST failed for {url}: {last_exc}")

def _last_n(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return rows[-n:] if rows and n < len(rows) else rows

async def _rap_for_polygon(geojson_polygon: Dict[str, Any], year: Optional[int]) -> Dict[str, Any]:
    payload = {"aoi": geojson_polygon}
    if year:
        payload["year"] = int(year)
    try:
        return await _http_post_json(RAP_URL, payload)
    except Exception as e:
        print(f"[WARN] RAP call failed: {e}")
        return {}

# ============================================================
# ROUTES
# ============================================================
@app.get("/")
async def root():
    return {"ok": True, "service": "Fine Fuel Loading API", "version": app.version}

@app.get("/health")
async def health():
    return {"ok": True, "ts": int(time.time())}

@app.get("/psas")
async def psas(gacc: Optional[str] = Query(None)):
    """Return CA PSAs."""
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "true",
        "outSR": 4326,
    }
    data = await _http_get_json(NIFC_PSA_GEOJSON, params)
    feats = data.get("features", [])
    if not feats:
        raise HTTPException(status_code=502, detail="No PSA features returned")
    wanted_gaccs = {x.strip().upper() for x in gacc.split(",")} if gacc else None
    filtered = []
    for f in feats:
        props = f.get("properties", {})
        state_val = props.get("STATE") or props.get("ST") or props.get("State") or props.get("STATE_NAME")
        gacc_val = props.get("GACC") or props.get("gacc")
        if (state_val in ("CA", "California")) and (
            not wanted_gaccs or (gacc_val and gacc_val.upper() in wanted_gaccs)
        ):
            filtered.append(f)
    return {"type": "FeatureCollection", "features": filtered or feats}

@app.get("/ca_psa_herbaceous")
async def ca_psa_herbaceous(
    year: Optional[int] = Query(None),
    intervals: int = Query(1, ge=1, le=12),
    gacc: Optional[str] = Query(None),
    include_series: bool = Query(False),
):
    """Return RAP herbaceous production by PSA."""
    cache_key = f"psa:{gacc}|year:{year}|int:{intervals}|series:{include_series}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    psa_fc = await psas(gacc)
    feats = psa_fc["features"]
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(f: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            return None
        geom = _thin_polygon(geom, 2000)

        async with sem:
            rap_json = await _rap_for_polygon(geom, year)

        series = _last_n(rap_json.get("production16day", []), intervals)
        rec = {
            "PSA_ID": props.get("PSA_ID") or props.get("PSAID"),
            "PSA_NAME": props.get("PSA_NAME") or props.get("PSANAME"),
            "GACC": props.get("GACC") or props.get("gacc"),
            "STATE": props.get("STATE") or props.get("ST") or props.get("STATE_NAME") or props.get("State"),
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

        return {"type": "Feature", "geometry": geom, "properties": rec}

    results = await asyncio.gather(*(worker(f) for f in feats))
    out_features = [r for r in results if r]
    out = {"type": "FeatureCollection", "features": out_features}
    cache_set(cache_key, out)
    return out

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
    intervals: int = Query(1, ge=1, le=12),
    metric: str = Query("HER", pattern="^(HER|AFG|PFG)$"),
    gacc: str = Query("OSCC"),
):
    """Count PSAs above threshold and return PL points."""
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
