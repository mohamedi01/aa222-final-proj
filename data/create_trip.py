#!/usr/bin/env python3
"""create_trip.py  –  Generate mock CSVs for an EV road‑trip (Stanford → USC)
without hammering the OpenRouteService API.

Key changes to avoid rate‑limit stalls:
──────────────────────────────────────
1. **Single directions call** to ORS.  We pull the full polyline once, then
   compute segment distances locally with the haversine formula – no more 40
   per‑segment calls.
2. **Exponential back‑off** wrapper around any ORS or NREL request: waits and
   retries instead of spamming warnings.
3. `NUM_SEGMENTS` defaults to 40 but can be passed via `$NUM_SEGMENTS` env.

Outputs (in script directory):
    route_segments.csv
    charging_stations.csv
    vehicle_params.csv
"""

from __future__ import annotations
import os, math, time, random
from pathlib import Path
from typing import List, Tuple

import requests
import openrouteservice as ors
import pandas as pd
import numpy as np
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────────
ORS_API_KEY  = os.getenv("ORS_API_KEY", "5b3ce3597851110001cf62484616ffaedd464af7b1ebcf23f22eb681")
NREL_API_KEY = os.getenv("NREL_API_KEY")  # optional
NUM_SEGMENTS = int(os.getenv("NUM_SEGMENTS", 40))

START_COORDS = (-122.1697, 37.4275)  # Stanford (lon, lat)
END_COORDS   = (-118.2851, 34.0224)  # USC      (lon, lat)

MOCK_PRICES = (0.25, 0.45)
MOCK_RATES  = (50, 250)

VEHICLE_PARAMS = {
    "Initial_Battery_kWh": 75,
    "Battery_Capacity_kWh": 100,
    "Min_Battery_kWh": 10,
    "Efficiency_kWh_per_mi": 0.30,
}

out_dir = Path(__file__).parent

# ────────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def haversine_mi(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Great‑circle distance in miles between two (lon, lat) points."""
    lon1, lat1, lon2, lat2 = map(math.radians, [*p1, *p2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def retry_request(func, *args, max_tries=5, **kwargs):
    delay = 1.0
    for attempt in range(max_tries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_tries - 1:
                raise
            time.sleep(delay)
            delay *= 2

# ────────────────────────────────────────────────────────────────────────────────
# ROUTE & SEGMENTS
# ────────────────────────────────────────────────────────────────────────────────

def get_polyline(client: ors.Client):
    print("Requesting full route once …")
    data = retry_request(
        client.directions,
        coordinates=[START_COORDS, END_COORDS],
        profile="driving-car",
        format="geojson",
    )
    return data["features"][0]["geometry"]["coordinates"]  # list [lon, lat]


def build_segments(coords: List[List[float]]) -> pd.DataFrame:
    total_dist = sum(haversine_mi(coords[i], coords[i+1]) for i in range(len(coords)-1))
    target_mi  = total_dist / NUM_SEGMENTS

    segments = []
    seg_start_idx, seg_id, acc = 0, 1, 0.0

    for i in range(1, len(coords)):
        step_dist = haversine_mi(coords[i-1], coords[i])
        acc += step_dist
        if acc >= target_mi or i == len(coords)-1:
            segments.append({
                "Segment_ID": seg_id,
                "Distance_mi": round(acc, 2),
                "Speed_limit": random.choice([55, 65, 40]),
                "Elevation_Change_ft": random.randint(-300, 300),
                "Avg_Temp_F": round(random.uniform(40, 80), 1),
            })
            seg_id += 1
            acc = 0.0
    return pd.DataFrame(segments)

# ────────────────────────────────────────────────────────────────────────────────
# CHARGERS (real via NREL or mock)
# ────────────────────────────────────────────────────────────────────────────────

def nrel_stations(lat: float, lon: float, radius=5) -> List[dict]:
    if not NREL_API_KEY:
        return []
    url = (
        f"https://developer.nrel.gov/api/alt-fuel-stations/v1.json?fuel_type=ELEC" \
        f"&api_key={NREL_API_KEY}&latitude={lat}&longitude={lon}&radius={radius}" \
        "&status=E&ev_charging_levels=dc_fast"
    )
    try:
        return retry_request(requests.get, url, timeout=10).json().get("fuel_stations", [])
    except Exception:
        return []


def build_chargers(segments_df: pd.DataFrame, coords: List[List[float]]) -> pd.DataFrame:
    chargers, idx = [], 0
    pts_per_seg = max(1, len(coords)//len(segments_df))

    for seg in segments_df.itertuples():
        lat, lon = coords[(seg.Segment_ID-1)*pts_per_seg][:2][::-1]
        real = nrel_stations(lat, lon)
        if real:
            for st in real:
                chargers.append({
                    "Segment_ID": seg.Segment_ID,
                    "Station_Name": st["station_name"][:40] + f"_{idx}",
                    "Price_per_kWh": float(st.get("ev_pricing", "0").lstrip("$") or random.uniform(*MOCK_PRICES)),
                    "Max_Charge_Rate_kW": int(max(st.get("ev_dc_fast_num", 1), 1) * 50),
                })
                idx += 1
        else:
            for _ in range(random.randint(0, 2)):
                chargers.append({
                    "Segment_ID": seg.Segment_ID,
                    "Station_Name": f"Mock_{seg.Segment_ID}_{idx}",
                    "Price_per_kWh": round(random.uniform(*MOCK_PRICES), 2),
                    "Max_Charge_Rate_kW": random.choice([50, 120, 150, 250]),
                })
                idx += 1
    return pd.DataFrame(chargers)

# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────

def main():
    client = ors.Client(key=ORS_API_KEY)

    coords = get_polyline(client)
    seg_df = build_segments(coords)
    seg_df.to_csv(out_dir / "route_segments.csv", index=False)

    st_df = build_chargers(seg_df, coords)
    st_df.to_csv(out_dir / "charging_stations.csv", index=False)

    pd.DataFrame({"Param": VEHICLE_PARAMS.keys(), "Value": VEHICLE_PARAMS.values()}).to_csv(out_dir / "vehicle_params.csv", index=False)

    print("✅  Generated CSVs in", out_dir)


if __name__ == "__main__":
    main()
