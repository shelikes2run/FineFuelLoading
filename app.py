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
import math
from typing import List, Dict, Any, Optional

import pandas as pd
import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# --------- Config from env ---------
EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT", "")
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "/etc/secrets/ee-key.json")
EE_COLLECTION_16D = os.getenv(
    "EE_COLLECTION_16D_PROV",
    "projects/rap-data-365417/assets/npp-partitioned-16day-v3-provisional",
)
PSA_NORMALS_CSV = os.getenv("PSA_NORMALS_CSV", "psa_HER_norm_CA_v3.csv")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

ARCGIS_PSA_URL = (
    "https://services3.arcgis.com/TA0MspDblLP3tGMV/arcgis/rest/services/"
    "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query"
)

# --------- Earth Engine init ---------
import ee  # type: ignore

def init_ee() -> Dict[str, Any]:
    ok = False
    msg = ""
    try:
        if EE_SERVICE_ACCOUNT and os.path.exists(EE_PRIVATE_KEY_FILE):
            creds = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_FILE)
            ee.Initialize(creds)
        else:
            ee.Initialize()
        ok = True
        msg = "EE initialized"
    except Exception as e:
        ok = False
        msg = f"EE init failed: {e}"
    return {"ok": ok, "message": msg}

EE_STATE = init_ee()

# --------- Load normals CSV once ---------
# Expected columns: PSANAME, GACCUnitID, afgNPP_norm, pfgNPP_norm, HER_norm
try:
    NORMALS = pd.read_csv(PSA_NORMALS_CSV).fillna("")
except Exception:
    NORMALS = pd.DataFrame(columns=["PSANAME","GACCUnitID","afgNPP_norm","pfgNPP_norm","HER_norm"])

# Helper: normalize GACC input
def parse_gaccs(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    raw = raw.replace(";", ",")
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    # Only two valid for CA, but keep generic
    return list(dict.fromkeys(vals))

# --------- ArcGIS fetch (robust) ---------
async def fetch_psa_features(gaccs: List[str]) -> Dict[str, Any]:
    if not gaccs:
        return {"features": [], "error": None}

    where = "GACCUnitID IN ({})".format(
        ",".join([f"'{g}'" for g in gaccs])
    )

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        # attempt 1: f=json, explicit outFields
        params = {
            "where": where,
            "returnGeometry": "true",
            "outFields": "OBJECTID,PSANAME,GACCUnitID,PSANationalCode",
            "outSR": 4326,
            "f": "json",
        }
        try:
            r = await client.get(ARCGIS_PSA_URL, params=params)
            r.raise_for_status()
            data = r.json()
            # Convert ArcGIS JSON to GeoJSON FeatureCollection
            feats = []
            for feat in data.get("features", []):
                attrs = feat.get("attributes", {}) or {}
                geom = feat.get("geometry", {}) or {}
                # polygon rings -> GeoJSON polygon
                if "rings" in geom:
                    coordinates = geom["rings"]
                    gj = {
                        "type": "Feature",
                        "properties": attrs,
                        "geometry": {"type": "Polygon", "coordinates": coordinates},
                    }
                    feats.append(gj)
            return {"features": feats, "error": None}
        except Exception as e1:
            err1 = str(e1)

        # attempt 2: f=geojson
        params2 = {
            "where": where,
            "returnGeometry": "true",
            "outFields": "*",
            "outSR": 4326,
            "f": "geojson",
        }
        try:
            r2 = await client.get(ARCGIS_PSA_URL, params=params2)
            r2.raise_for_status()
            data2 = r2.json()
            feats2 = data2.get("features", [])
            return {"features": feats2, "error": None}
        except Exception as e2:
            return {"features": [], "error": f"ArcGIS fetch failed: {err1} | {e2}"}

# --------- EE: latest composite + zonal means ---------
def ee_latest_image_and_date(coll_id: str) -> Dict[str, Any]:
    try:
        coll = ee.ImageCollection(coll_id).sort("system:time_start", False)
        img = coll.first()
        info = img.getInfo()
        ts = info["properties"]["system:time_start"]
        date = ee.Date(ts).format("YYYY-MM-dd").getInfo()
        return {"image": ee.Image(img), "date": date, "error": None}
    except Exception as e:
        return {"image": None, "date": None, "error": str(e)}

def ee_reduce_means(img: ee.Image, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Prepare EE FeatureCollection carrying PSA props we care about
    ee_feats = []
    for f in features:
        props = f.get("properties", {}) or {}
        geom  = f.get("geometry", {}) or {}
        if not geom:
            continue
        try:
            ee_geom = ee.Geometry(geom)
            ee_feat = ee.Feature(ee_geom, {
                "PSANAME": props.get("PSANAME",""),
                "GACCUnitID": props.get("GACCUnitID",""),
                "PSANationalCode": props.get("PSANationalCode",""),
            })
            ee_feats.append(ee_feat)
        except Exception:
            # skip broken geometry
            continue

    if not ee_feats:
        return []

    fc = ee.FeatureCollection(ee_feats)

    # Bands we care about (NPP partitioned: afgNPP, pfgNPP, HER)
    bands = []
    for b in ["afgNPP", "pfgNPP", "HER"]:
        try:
            # only include if present
            _ = img.select(b)
            bands.append(b)
        except Exception:
            pass
    if not bands:
        return []

    red = img.select(bands).reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30  # RAP 16-day provisional is 30 m
    )

    out = red.getInfo().get("features", [])
    rows = []
    for f in out:
        p = f.get("properties", {}) or {}
        rows.append({
            "PSANAME": p.get("PSANAME",""),
            "GACCUnitID": p.get("GACCUnitID",""),
            "PSANationalCode": p.get("PSANationalCode",""),
            "afgNPP_latest": p.get("afgNPP_mean"),
            "pfgNPP_latest": p.get("pfgNPP_mean"),
            "HER_latest":   p.get("HER_mean"),
        })
    return rows

# --------- Merge with normals, compute flags ---------
def merge_with_normals(latest_rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not latest_rows:
        return []

    latest = pd.DataFrame(latest_rows)
    # Left-join to your CSV by PSANAME; if that ever misses, try PSANationalCode
    m = latest.merge(NORMALS, on=["PSANAME","GACCUnitID"], how="left")

    # Flag above normal (HER as primary, but keep per-part too)
    def gt(a, b):
        try:
            return float(a) > float(b)
        except Exception:
            return None

    out = []
    for _, r in m.iterrows():
        out.append({
            "PSANAME": r.get("PSANAME",""),
            "PSANationalCode": r.get("PSANationalCode",""),
            "GACCUnitID": r.get("GACCUnitID",""),
            "afgNPP_latest": r.get("afgNPP_latest"),
            "pfgNPP_latest": r.get("pfgNPP_latest"),
            "HER_latest":    r.get("HER_latest"),
            "afgNPP_norm":   r.get("afgNPP_norm"),
            "pfgNPP_norm":   r.get("pfgNPP_norm"),
            "HER_norm":      r.get("HER_norm"),
            "above_normal":  gt(r.get("HER_latest"), r.get("HER_norm")),
        })
    return out

# --------- FastAPI ---------
app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ee_initialized": EE_STATE["ok"],
        "normals_csv": PSA_NORMALS_CSV,
    }

@app.get("/psa_flags")
async def psa_flags(
    gaccs: Optional[str] = Query(None, description="Comma or semicolon separated GACC IDs (e.g. USCAOSCC,USCAONCC)"),
    pretty: Optional[int] = Query(0)
):
    if not EE_STATE["ok"]:
        return JSONResponse(status_code=500, content={"detail": EE_STATE["message"]})

    gacc_list = parse_gaccs(gaccs)
    # ArcGIS: get PSA polygons + IDs
    ag = await fetch_psa_features(gacc_list)

    # EE: find latest composite and zonal means
    latest_info = ee_latest_image_and_date(EE_COLLECTION_16D)
    payload: Dict[str, Any] = {
        "count": 0,
        "gaccs": gacc_list,
        "collection": EE_COLLECTION_16D,
        "latest_composite": latest_info["date"],
        "rows": [],
    }

    if ag["error"]:
        payload["error"] = ag["error"]

    rows = []
    if (latest_info["image"] is not None) and ag["features"]:
        rows = ee_reduce_means(latest_info["image"], ag["features"])
        merged = merge_with_normals(rows)
        payload["rows"] = merged
        payload["count"] = len(merged)

    # pretty-print switch
    if pretty:
        return JSONResponse(
            content=json.loads(json.dumps(payload, indent=2, allow_nan=False)),
            media_type="application/json",
        )
    else:
        # ensure strict JSON (no NaN/inf)
        return JSONResponse(
            content=json.loads(json.dumps(payload, allow_nan=False)),
            media_type="application/json",
        )
