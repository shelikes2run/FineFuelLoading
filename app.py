import os
import time
import json as _json
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
NIFC_PSA_GEOJSON = os.getenv(
    "NIFC_PSA_GEOJSON",
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

# RAP 10m vegetation cover (annual), community catalog
EE_COLLECTION_10M = os.getenv(
    "EE_COLLECTION_10M",
    "projects/rap-data-365417/assets/vegetation-cover-10m"
)

# Runtime knobs (keep memory tiny on Render Starter)
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "60"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "3600"))
MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "6"))

# ---------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------
app = FastAPI(title="CA PSA Herbaceous API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Simple capped in-memory cache
# ---------------------------------------------------------------------
_cache: Dict[str, Tuple[float, Any]] = {}

def cache_get(key: str):
    rec = _cache.get(key)
    if not rec:
        return None
    exp, val = rec
    if time.time() > exp:
        _cache.pop(key, None)
        return None
    return val

def cache_set(key: str, val: Any, ttl: int = CACHE_TTL_SEC):
    if len(_cache) >= MAX_CACHE_ENTRIES:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)
    _cache[key] = (time.time() + ttl, val)

# ---------------------------------------------------------------------
# Single shared HTTP client
# ---------------------------------------------------------------------
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

async def _http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = await _client.get(url, params=params)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _pl_points_from_count(affected_count: int) -> int:
    """SouthOps PL scale based on # of PSAs affected."""
    if affected_count <= 1: return 2
    if affected_count <= 4: return 4
    if affected_count <= 7: return 6
    if affected_count <= 10: return 8
    return 10

def _thin_polygon(geom: Dict[str, Any], max_points: int = 2000) -> Dict[str, Any]:
    """Downsample polygon rings to keep AOIs lightweight."""
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

    t = geom.get("type")
    if t == "Polygon":
        coords = geom.get("coordinates", [])
        if not coords:
            return geom
        shell = _thin_ring(coords[0], max_points)
        holes = [_thin_ring(r, max_points // 5) for r in coords[1:]]
        return {"type": "Polygon", "coordinates": [shell] + holes}

    if t == "MultiPolygon":
        polys = []
        for poly in geom.get("coordinates", []):
            if not poly:
                continue
            shell = _thin_ring(poly[0], max_points)
            holes = [_thin_ring(r, max_points // 5) for r in poly[1:]]
            polys.append([shell] + holes)
        if not polys:
            return {"type": "Polygon", "coordinates": []}
        # collapse to first polygon to keep payload small
        return {"type": "Polygon", "coordinates": polys[0]}

    return geom

# ---------------------------------------------------------------------
# Google Earth Engine init
# ---------------------------------------------------------------------
EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")       # <svc>@<project>.iam.gserviceaccount.com
EE_PRIVATE_KEY_JSON = os.getenv("EE_PRIVATE_KEY_JSON")     # Full JSON key (string)

_ee_initialized = False

def _gee_init():
    """Initialize Earth Engine once per process (service account preferred)."""
    global _ee_initialized
    if _ee_initialized:
        return
    try:
        import ee
        if EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_JSON:
            creds = ee.ServiceAccountCredentials(
                EE_SERVICE_ACCOUNT, key_data=_json.loads(EE_PRIVATE_KEY_JSON)
            )
            ee.Initialize(creds)
        else:
            ee.Initialize()  # local dev fallback
        _ee_initialized = True
        print("[GEE] Initialized")
    except Exception as e:
        print("[GEE] Init failed:", e)
        raise HTTPException(status_code=500, detail=f"GEE init failed: {e}")

def _geojson_to_ee_geometry(geom: Dict[str, Any]):
    import ee
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not gtype or coords is None:
        raise HTTPException(400, "Invalid geometry")
    if gtype == "Polygon":
        return ee.Geometry.Polygon(coords, None, False)
    if gtype == "MultiPolygon":
        return ee.Geometry.MultiPolygon(coords, None, False)
    raise HTTPException(400, f"Unsupported geometry type: {gtype}")

# ---------------------------------------------------------------------
# Core: current vs. normal (2018â€“2023) for RAP 10m AFG+PFG
# ---------------------------------------------------------------------
async def _gee_latest_stats_for_geom(geom: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Compute mean Annual & Perennial Forb/Grass cover (%) for AOI using RAP 10 m.
    Returns latest-year value + 2018â€“2023 normal + % difference and AboveNormal flag.
    """
    def _work():
        import ee
        _gee_init()

        ic = ee.ImageCollection(EE_COLLECTION_10M)
        ee_geom = _geojson_to_ee_geometry(geom)

        # Historical normal across 2018â€“2023 inclusive
        hist_ic = ic.filterDate("2018-01-01", "2024-01-01")
        latest = ic.sort("system:time_start", False).first()

        if latest is None:
            return None

        # Bands of interest
        bands = ["AnnualForbGrass", "PerennialForbGrass"]

        # Normal (2018â€“2023) mean within AOI
        hist_mean = hist_ic.select(bands).mean().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_geom,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        )

        # Latest year mean within AOI
        latest_stats = latest.select(bands).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_geom,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        )

        hist = hist_mean.getInfo() or {}
        curr = latest_stats.getInfo() or {}
        year_str = ee.Date(latest.get("system:time_start")).format("YYYY").getInfo()

        afg = curr.get("AnnualForbGrass")
        pfg = curr.get("PerennialForbGrass")
        afg_norm = hist.get("AnnualForbGrass")
        pfg_norm = hist.get("PerennialForbGrass")

        curr_total = (afg or 0.0) + (pfg or 0.0)
        norm_total = (afg_norm or 0.0) + (pfg_norm or 0.0)

        pct_diff = None
        if norm_total:
            pct_diff = 100.0 * (curr_total - norm_total) / norm_total

        return {
            "year": year_str,
            "AFG": afg,
            "PFG": pfg,
            "AFG_norm": afg_norm,
            "PFG_norm": pfg_norm,
            "Total_current": curr_total,
            "Total_normal": norm_total,
            "PctDiff": pct_diff,
            "AboveNormal": (pct_diff is not None and pct_diff > 0.0)
        }

    return await asyncio.to_thread(_work)

# ---------------------------------------------------------------------
# Routes: basic / health
# ---------------------------------------------------------------------
@app.head("/")
@app.get("/")
async def root():
    return {"ok": True, "service": "CA PSA Herbaceous API", "version": app.version}

@app.get("/health")
async def health():
    return {"ok": True, "ts": int(time.time())}

# ---------------------------------------------------------------------
# Routes: PSAs (GeoJSON)
# ---------------------------------------------------------------------
@app.get("/psas")
async def psas(gacc: Optional[str] = Query(None, description="OSCC, ONCC, or both comma-separated")):
    """
    Return CA PSA polygons as GeoJSON FeatureCollection (filtered to OSCC/ONCC if provided).
    """
    cache_key = f"psa_base:{gacc}"
    cached = cache_get(cache_key)
    if cached:
        return cached

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
        raise HTTPException(502, "No PSA features returned")

    wanted = {x.strip().upper() for x in gacc.split(",")} if gacc else None
    filtered = []
    for f in feats:
        props = f.get("properties", {}) or {}
        st = props.get("STATE") or props.get("ST") or props.get("STATE_NAME") or props.get("State")
        g = props.get("GACC") or props.get("gacc")
        if (st in ("CA", "California")) and (not wanted or (g and g.upper() in wanted)):
            filtered.append(f)

    out = {"type": "FeatureCollection", "features": filtered or feats}
    cache_set(cache_key, out)
    return out

# ---------------------------------------------------------------------
# Routes: RAP 10m per-PSA + PL points (AboveNormal)
# ---------------------------------------------------------------------
@app.get("/ca_psa_herbaceous_gee")
async def ca_psa_herbaceous_gee(
    gacc: Optional[str] = Query(None, description="OSCC, ONCC, or both comma-separated"),
):
    """
    GeoJSON of PSAs with RAP 10 m vegetation cover stats:
      - AFG, PFG (latest year % cover)
      - AFG_norm, PFG_norm (2018â€“2023 normal)
      - Total_current, Total_normal, PctDiff, AboveNormal (AFG+PFG combined)
      - Year_latest
    """
    cache_key = f"gee10m_fc:{gacc}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    psa_fc = await psas(gacc=gacc)
    feats = psa_fc.get("features", [])
    if not feats:
        out = {"type": "FeatureCollection", "features": []}
        cache_set(cache_key, out, ttl=600)
        return out

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(f: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        props = f.get("properties", {})
        geom = f.get("geometry", {})
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            return None
        g = _thin_polygon(geom, 1500)
        async with sem:
            rec = await _gee_latest_stats_for_geom(g)

        outp = {
            "PSA_ID":   props.get("PSA_ID") or props.get("PSAID"),
            "PSA_NAME": props.get("PSA_NAME") or props.get("PSANAME"),
            "GACC":     props.get("GACC") or props.get("gacc"),
            "STATE":    props.get("STATE") or props.get("ST") or props.get("STATE_NAME") or props.get("State"),
        }
        if rec:
            outp.update({
                "AFG_latest": rec.get("AFG"),
                "PFG_latest": rec.get("PFG"),
                "AFG_norm": rec.get("AFG_norm"),
                "PFG_norm": rec.get("PFG_norm"),
                "HER_total_latest": rec.get("Total_current"),
                "HER_total_normal": rec.get("Total_normal"),
                "HER_pct_diff": rec.get("PctDiff"),
                "HER_above_normal": rec.get("AboveNormal"),
                "Year_latest": rec.get("year"),
            })
        return {"type": "Feature", "geometry": g, "properties": outp}

    results = await asyncio.gather(*(worker(f) for f in feats))
    out_fc = {"type": "FeatureCollection", "features": [r for r in results if r]}
    cache_set(cache_key, out_fc)
    return out_fc

@app.get("/pl_abovenormal_points_gee")
async def pl_abovenormal_points_gee(
    gacc: str = Query("OSCC", description="OSCC, ONCC, or ONCC,OSCC"),
):
    """
    PL points using RAP 10 m vegetation cover:
      - Count PSAs where (AFG+PFG) latest > normal (2018â€“2023)
      - Map count â†’ SouthOps scale (2,4,6,8,10)
    """
    fc = await ca_psa_herbaceous_gee(gacc=gacc)
    feats = fc.get("features", [])

    affected = 0
    for f in feats:
        p = f["properties"]
        if p.get("HER_above_normal") is True:
            affected += 1

    return {
        "gacc": gacc,
        "metric": "HER_cover_above_normal",
        "total_psas": len(feats),
        "affected_count": affected,
        "points": _pl_points_from_count(affected),
        "as_of_year": feats[0]["properties"].get("Year_latest") if feats else None
    }

# Convenience aliases
@app.get("/southops_pl_points_gee")
async def southops_pl_points_gee():
    return await pl_abovenormal_points_gee(gacc="OSCC")

@app.get("/northops_pl_points_gee")
async def northops_pl_points_gee():
    return await pl_abovenormal_points_gee(gacc="ONCC")
