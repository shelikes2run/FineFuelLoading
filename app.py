# app.py
# CA PSA Herbaceous (HER) API — uses RAP 16-day provisional (afgNPP/pfgNPP) vs PSA normals CSV
# Render env vars expected:
#   EE_SERVICE_ACCOUNT        (e.g., finefuel@finefuelloading.iam.gserviceaccount.com)
#   EE_PRIVATE_KEY_FILE       (/etc/secrets/ee-key.json)
#   PSA_NORMALS_CSV           (default: psa_HER_norm_CA_v3.csv)
#   EE_COLLECTION_16D_PROV    (default: projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional)
#   CACHE_TTL                 (seconds; default 900)

# app.py
import os
import json
import time
import re
from functools import lru_cache
from typing import List, Dict, Any

import httpx
import pandas as pd
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware

import ee

# -------------------------
# Config (env overrides)
# -------------------------
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT_SEC", "60"))

EE_COLLECTION_16D_PROV = os.environ.get(
    "EE_COLLECTION_16D_PROV",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional",
)

EE_SERVICE_ACCOUNT = os.environ.get("EE_SERVICE_ACCOUNT", "")
EE_PRIVATE_KEY_FILE = os.environ.get("EE_PRIVATE_KEY_FILE", "")

PSA_CSV = os.environ.get("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")

ARCGIS_PSA_URL = (
    "https://services3.arcgis.com/1A0MspDfblp3GfNY/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

# -------------------------
# App
# -------------------------
app = FastAPI(title="Fine Fuel Loading â€“ PSA Flags")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Utilities
# -------------------------
def _name_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _safe_float(x) -> float:
    try:
        if x is None:
            return 0.0
        v = float(x)
        if v != v:  # NaN check
            return 0.0
        return v
    except Exception:
        return 0.0

@lru_cache(maxsize=1)
def load_norms_df() -> pd.DataFrame:
    """Load PSA normals CSV (authoritative for names + codes)."""
    df = pd.read_csv(PSA_CSV).fillna("")
    # Expect these columns:
    required = {
        "PSANationalCode", "PSANAME", "GACCUnitID",
        "afgNPP_norm", "pfgNPP_norm", "HER_norm",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing columns: {sorted(missing)}")

    # Helper keys for robust fallback
    df["__key"] = df["PSANationalCode"].astype(str).str.strip()
    df["__norm_name"] = df["PSANAME"].map(_name_norm)
    return df

def init_ee() -> bool:
    """Initialize Earth Engine with a service account."""
    try:
        if EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE:
            credentials = ee.ServiceAccountCredentials(
                EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_FILE
            )
            ee.Initialize(credentials)
        else:
            # fallback to whatever default is available (rare in Render)
            ee.Initialize()
        return True
    except Exception:
        return False

EE_READY = init_ee()

def latest_composite_info() -> Dict[str, Any]:
    """Return latest image and its YYYY-MM-DD string from the 16-day collection."""
    col = ee.ImageCollection(EE_COLLECTION_16D_PROV)
    latest = col.sort("system:time_start", False).first()
    date_str = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    return {"image": latest, "date": date_str}

def fetch_psa_features(gacc_list: List[str]) -> List[Dict[str, Any]]:
    """Fetch PSA polygons from ArcGIS filtered by GACCUnitID list."""
    where = "1=1"
    if gacc_list:
        quoted = ",".join([f"'{g.strip()}'" for g in gacc_list if g.strip()])
        where = f"GACCUnitID IN ({quoted})"

    params = {
        "f": "geojson",
        "where": where,
        "returnGeometry": "true",
        "outFields": "OBJECTID,PSANAME,PSANationalCode,GACCUnitID",
        "outSR": 4326,
        "resultRecordCount": 5000,
    }
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        r = client.get(ARCGIS_PSA_URL, params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("features", [])

def join_polys_with_norms(features: List[Dict[str, Any]]) -> pd.DataFrame:
    """Join ArcGIS features to CSV by PSANationalCode (fallback: name)."""
    norms = load_norms_df()

    rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        code = str(props.get("PSANationalCode", "") or "").strip()
        name = (props.get("PSANAME") or "").strip()
        gacc = (props.get("GACCUnitID") or "").strip()

        match = None
        if code:
            m = norms[norms["__key"] == code]
            if len(m):
                match = m.iloc[0]

        if match is None and name:
            nn = _name_norm(name)
            m = norms[norms["__norm_name"] == nn]
            if len(m):
                match = m.iloc[0]

        if match is None:
            # Skip unmapped features (keeps output clean)
            continue

        rows.append({
            "PSANationalCode": match["PSANationalCode"],
            "PSANAME": match["PSANAME"],
            "GACCUnitID": match["GACCUnitID"] or gacc,
            "afgNPP_norm": _safe_float(match.get("afgNPP_norm")),
            "pfgNPP_norm": _safe_float(match.get("pfgNPP_norm")),
            "HER_norm": _safe_float(match.get("HER_norm")),
            "_geometry": f.get("geometry"),
        })

    return pd.DataFrame(rows)

def compute_latest_by_polygons(df: pd.DataFrame) -> pd.DataFrame:
    """Sample the latest 16-day composite over PSA polygons."""
    if df.empty:
        return df.assign(afgNPP_latest=0.0, pfgNPP_latest=0.0, HER_latest=0.0)

    info = latest_composite_info()
    img = info["image"]  # ee.Image
    # Use only afgNPP / pfgNPP bands
    img = img.select(["afgNPP", "pfgNPP"])

    # Build a FeatureCollection from the CSV rows with geometry from ArcGIS
    feats = []
    for _, row in df.iterrows():
        geom = ee.Geometry(row["_geometry"])
        props = {
            "PSANationalCode": row["PSANationalCode"],
            "PSANAME": row["PSANAME"],
            "GACCUnitID": row["GACCUnitID"],
            "afgNPP_norm": row["afgNPP_norm"],
            "pfgNPP_norm": row["pfgNPP_norm"],
            "HER_norm": row["HER_norm"],
        }
        feats.append(ee.Feature(geom, props))

    fc = ee.FeatureCollection(feats)

    # reduceRegions supports: reducer, collection, scale, crs, crsTransform, tileScale
    # (No bestEffort / maxPixels here.)
    reduced = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=2,
    )

    # Pull results back
    out = reduced.getInfo()  # small (~30 features), okay to getInfo
    features = out.get("features", [])

    records = []
    for f in features:
        p = f.get("properties", {}) or {}
        afg = _safe_float(p.get("afgNPP"))  # reducer stores band name as key
        pfg = _safe_float(p.get("pfgNPP"))
        her = afg + pfg

        rec = {
            "PSANationalCode": p.get("PSANationalCode", ""),
            "PSANAME": p.get("PSANAME", ""),
            "GACCUnitID": p.get("GACCUnitID", ""),
            "afgNPP_latest": round(afg, 6),
            "pfgNPP_latest": round(pfg, 6),
            "HER_latest": round(her, 6),
            "afgNPP_norm": round(_safe_float(p.get("afgNPP_norm")), 6),
            "pfgNPP_norm": round(_safe_float(p.get("pfgNPP_norm")), 6),
            "HER_norm": round(_safe_float(p.get("HER_norm")), 6),
        }
        rec["above_normal"] = 1 if rec["HER_latest"] > rec["HER_norm"] else 0
        records.append(rec)

    # Return as DataFrame
    return pd.DataFrame.from_records(records)

def json_response(payload: Dict[str, Any], pretty: bool) -> Response:
    # Ensure there are no NaNs for strict JSON
    def _clean(o):
        if isinstance(o, float):
            return 0.0 if (o != o) else o
        if isinstance(o, list):
            return [_clean(x) for x in o]
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        return o

    cleaned = _clean(payload)
    text = json.dumps(cleaned, indent=2 if pretty else None)
    return Response(content=text, media_type="application/json")

# -------------------------
# Routes
# -------------------------
@app.get("/")
def health():
    return {"status": "ok", "ee_initialized": EE_READY, "normals_csv": os.path.basename(PSA_CSV)}

@app.get("/psa_flags")
def psa_flags(
    gaccs: str = Query("", description="Comma-separated GACCUnitID values (e.g., USCAOSCC,USCAONCC)"),
    pretty: int = Query(0, description="Pretty-print JSON if 1"),
):
    if not EE_READY:
        return json_response(
            {"count": 0, "gaccs": gaccs.split(",") if gaccs else [], "error": "Earth Engine not initialized"},
            bool(pretty),
        )

    # Parse/normalize GACC filters
    gacc_list = [g.strip() for g in gaccs.split(",") if g.strip()] if gaccs else []

    # Fetch polygons from ArcGIS & join with CSV normals
    try:
        feats = fetch_psa_features(gacc_list)
    except Exception as e:
        return json_response(
            {"count": 0, "gaccs": gacc_list, "error": f"ArcGIS fetch failed: {str(e)}"},
            bool(pretty),
        )

    df_joined = join_polys_with_norms(feats)

    # If nothing matched (e.g., CSV/GACC mismatch), return empty
    if df_joined.empty:
        return json_response(
            {"count": 0, "gaccs": gacc_list, "collection": EE_COLLECTION_16D_PROV, "latest_composite": None, "rows": []},
            bool(pretty),
        )

    # Compute latest
    try:
        latest = latest_composite_info()
        latest_date = latest["date"]
    except Exception as e:
        return json_response(
            {"count": 0, "gaccs": gacc_list, "collection": EE_COLLECTION_16D_PROV, "error": f"Latest composite error: {str(e)}"},
            bool(pretty),
        )

    # Sample image over polygons
    try:
        df_vals = compute_latest_by_polygons(df_joined)
    except Exception as e:
        return json_response(
            {"count": 0, "gaccs": gacc_list, "collection": EE_COLLECTION_16D_PROV, "latest_composite": latest_date,
             "error": f"EE reduce error: {str(e)}"},
            bool(pretty),
        )

    rows = df_vals.to_dict(orient="records")
    payload = {
        "count": len(rows),
        "gaccs": gacc_list,
        "collection": EE_COLLECTION_16D_PROV,
        "latest_composite": latest_date,
        "rows": rows,
    }
    return json_response(payload, bool(pretty))


# -------------------------
# Local dev entry
# -------------------------
if __name__ == "__main__":
    # For local testing only
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), reload=False)
