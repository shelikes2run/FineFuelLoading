import os
import time
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# CONFIG (env overrides allowed)
# -----------------------------
NIFC_PSA_URL = os.getenv(
"NIFC_PSA_URL",
"https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/"
"Predictive_Service_Area_PSA_Boundaries_Public/FeatureServer/0/query"
)
RAP_URL = os.getenv(
"RAP_URL",
"https://us-central1-rap-data-365417.cloudfunctions.net/production16dayV3"
)
# Concurrency & timeouts
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "120"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "1.7"))

# Cache (seconds)
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "21600")) # 6 hours default

# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="CA PSA Herbaceous API", version="1.0.0")

# Allow dashboards/apps to call us directly
app.add_middleware(
CORSMiddleware,
allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# -------------
# In-mem cache
# -------------
_cache: Dict[str, Tuple[float, Any]] = {} # key -> (expires_at, value)
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

# -------------
# HTTP helpers
# -------------
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
await asyncio.sleep((RETRY_BACKOFF_BASE ** (attempt - 1)))
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
await asyncio.sleep((RETRY_BACKOFF_BASE ** (attempt - 1)))
raise HTTPException(status_code=502, detail=f"POST failed for {url}: {last_exc}")

# ---------- PL scoring helpers ----------
def _pl_points_from_count(affected_count: int) -> int:
"""
South Ops scale:
0–1 = 2 pts
2–4 = 4 pts
5–7 = 6 pts
8–10 = 8 pts
11+ = 10 pts
"""
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
intervals: int = Query(1, ge=1, le=12, description="How many most-recent 16-day intervals to fetch (we evaluate the latest)."),
metric: str = Query("HER", pattern="^(HER|AFG|PFG)$", description="Which RAP field to test against threshold: HER, AFG, or PFG."),
gacc: str = Query("OSCC", description="Which GACC(s) to evaluate. Default OSCC (South Ops). Use 'ONCC' or 'ONCC,OSCC' if desired."),
):
"""
Count PSAs in the chosen GACC(s) where <metric>_latest >= threshold_lbsac,
then map that count to PL points using South Ops scale.
"""
# Reuse the logic that builds PSA features with latest metrics (fast + cached)
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
f"{metric}_latest": val
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
"affected_psas": affected # handy for QA or popup drilldown
}

# ----------------------------
# ArcGIS FeatureService helpers
# ----------------------------
def _arcgis_query_params(where: str, f: str = "geojson", out_fields: str = "*", return_geometry: bool = True):
return {
"where": where,
"outFields": out_fields,
"f": f,
"returnGeometry": "true" if return_geometry else "false",
"outSR": 4326,
}

async def fetch_ca_psas(gacc: Optional[str] = None) -> Dict[str, Any]:
"""
Returns GeoJSON FeatureCollection of CA PSAs. Filtered to ONCC/OSCC by default.
"""
gacc_filter = ""
if gacc:
# single or comma-delimited list (normalize to tuple)
gaccs = [val.strip().upper() for val in gacc.split(",")]
quoted = ",".join([f"'{x}'" for x in gaccs])
gacc_filter = f" AND GACC IN ({quoted})"
else:
gacc_filter = " AND GACC IN ('ONCC','OSCC')"

where = "STATE = 'CA'" + gacc_filter
params = _arcgis_query_params(where=where, f="geojson", out_fields="PSA_ID,PSA_NAME,GACC,STATE", return_geometry=True)
data = await _http_get_json(NIFC_PSA_URL, params)
if "features" not in data:
raise HTTPException(status_code=502, detail="Unexpected PSA response")
return data

# -----------------------
# RAP helpers / utilities
# -----------------------
def _esri_geojson_to_geojson_polygon(geom: Dict[str, Any]) -> Dict[str, Any]:
"""
The NIFC endpoint already returns GeoJSON geometry; we just sanity-check Polygon type.
If it's MultiPolygon, keep first polygon shell+holes (rare for PSAs).
"""
gtype = geom.get("type")
if gtype == "Polygon":
return geom
if gtype == "MultiPolygon":
# collapse to first polygon
coords = geom.get("coordinates", [])
if not coords:
raise ValueError("Empty MultiPolygon")
return {"type": "Polygon", "coordinates": coords[0]}
raise ValueError(f"Unsupported geometry type: {gtype}")

def _last_n(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
if not rows:
return []
return rows[-n:] if n and n < len(rows) else rows

async def _rap_for_polygon(geojson_polygon: Dict[str, Any], year: Optional[int]) -> Dict[str, Any]:
payload = {"aoi": geojson_polygon}
if year:
payload["year"] = int(year)
return await _http_post_json(RAP_URL, payload)

# -----------------------
# API routes
# -----------------------
@app.get("/health")
async def health():
return {"ok": True, "ts": int(time.time())}

@app.get("/psas")
async def psas(gacc: Optional[str] = Query(None, description="Comma-delimited GACCs (e.g., 'ONCC,OSCC'). Default both.")):
"""
Raw CA PSA polygons (from NIFC) as GeoJSON FeatureCollection.
"""
return await fetch_ca_psas(gacc=gacc)

@app.get("/ca_psa_herbaceous")
async def ca_psa_herbaceous(
year: Optional[int] = Query(None, description="Optional filter year for RAP."),
intervals: int = Query(3, ge=1, le=12, description="How many most-recent 16-day intervals to include (1-12)."),
gacc: Optional[str] = Query(None, description="Comma-delimited GACCs (default ONCC,OSCC)."),
include_series: bool = Query(True, description="If false, only '..._latest' fields + interval date are returned."),
):
"""
Returns GeoJSON FeatureCollection with AFG/PFG/HER (lbs/acre) summarized per CA PSA
for the last N 16-day intervals, plus '..._latest' fields.
"""
cache_key = f"psa:{gacc}|year:{year}|int:{intervals}|series:{include_series}"
cached = cache_get(cache_key)
if cached:
return cached

psa_fc = await fetch_ca_psas(gacc=gacc)
feats = psa_fc["features"]

# Concurrency guard
sem = asyncio.Semaphore(MAX_CONCURRENCY)

async def worker(f: Dict[str, Any]) -> Optional[Dict[str, Any]]:
props = f.get("properties", {})
geom = f.get("geometry", {})
try:
poly = _esri_geojson_to_geojson_polygon(geom)
except Exception:
# skip bad geometry
return None

# RAP call (bounded concurrency)
async with sem:
rap_json = await _rap_for_polygon(poly, year)

series = rap_json.get("production16day", []) or []
series = _last_n(series, intervals)

# base attributes
psa_id = props.get("PSA_ID") or props.get("PSAID")
psa_name = props.get("PSA_NAME") or props.get("PSANAME")
rec = {
"PSA_ID": psa_id,
"PSA_NAME": psa_name,
"GACC": props.get("GACC"),
"STATE": props.get("STATE"),
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
# Keep short/clean; only add if date present
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
