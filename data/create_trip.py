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
import osmnx as ox   
import networkx as nx 

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


def build_segments(coords: List[List[float]], G_drive: nx.DiGraph) -> pd.DataFrame:
    total_dist = sum(haversine_mi(coords[i], coords[i+1]) for i in range(len(coords)-1))
    target_mi  = total_dist / NUM_SEGMENTS

    segments = []
    seg_start_idx, seg_id, acc = 0, 1, 0.0

    for i in range(1, len(coords)):
        step_dist = haversine_mi(coords[i-1], coords[i])
        acc += step_dist
        if acc >= target_mi or i == len(coords)-1:
            lon_mid, lat_mid = coords[i]
            temp_f = fetch_temperature_f(lat_mid, lon_mid)
            time.sleep(1)  # avoid hitting Open-Meteo too fast

            # Snap to nearest OSM edge to get actual posted speed limit
            u, v, key = ox.distance.nearest_edges(G_drive, lon_mid, lat_mid)
            speed_kph = G_drive[u][v][key].get("speed_kph", 40)  # fallback 40 kph if missing
            speed_mph = speed_kph * 0.621371

            # Real elevation at segment start & end
            lon_start, lat_start = coords[seg_start_idx]
            lon_end,   lat_end   = coords[i]

            elev_start_m = fetch_elevation_m(lat_start, lon_start)
            time.sleep(0.5)  # throttle Open-Elevation
            elev_end_m   = fetch_elevation_m(lat_end, lon_end)
            elev_change_ft = round((elev_end_m - elev_start_m) * 3.28084)

            segments.append({
                "Segment_ID": seg_id,
                "Distance_mi": round(acc, 2),
                "Speed_limit": round(speed_mph),
                "Elevation_Change_ft": elev_change_ft,
                "Avg_Temp_F": round(temp_f, 1),
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

def fetch_elevation_m(lat: float, lon: float) -> float:
    """
    Query the Open-Elevation API for elevation at (lat, lon) in meters.
    If the call fails, return 0.0 as a fallback.
    """
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat:.5f},{lon:.5f}"
        r = retry_request(requests.get, url, timeout=5)
        data = r.json()
        return float(data["results"][0]["elevation"])
    except Exception:
        return 0.0


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
# Temperature (real via Open-Meteo)
# ────────────────────────────────────────────────────────────────────────────────
def fetch_temperature_f(lat: float, lon: float) -> float:
    """
    Query Open-Meteo for the current temperature (°C) at (lat, lon),
    convert to °F, and return. If the request fails, fall back to 70°F.
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat:.5f}&longitude={lon:.5f}"
            "&current_weather=true&timezone=UTC"
        )
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        temp_c = data["current_weather"]["temperature"]
        return temp_c * 9/5 + 32  # convert to °F
    except Exception:
        return 70.0  # fallback if something goes wrong


# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────

def main():
    client = ors.Client(key=ORS_API_KEY)

    coords = get_polyline(client)


    # ────────────────────────────────────────────────────────────────────────────────
    # 1b) Build OSMnx “drive” graph over the route’s bounding box
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    north, south = max(lats) + 0.01, min(lats) - 0.01
    east,  west  = max(lons) + 0.01, min(lons) - 0.01

    print("Downloading OSM graph for bbox:", north, south, east, west)
    bbox = (north, south, east, west)
    G_drive = ox.graph_from_bbox(bbox, network_type="drive")
    G_drive = ox.add_edge_speeds(G_drive)
    G_drive = ox.add_edge_travel_times(G_drive)

    # ────────────────────────────────────────────────────────────────────────────────
    # 2) Build segments, using real temp + posted speed limit
    seg_df = build_segments(coords, G_drive)    # pass G_drive into build_segments
    seg_df.to_csv(out_dir / "route_segments.csv", index=False)

    st_df = build_chargers(seg_df, coords)
    st_df.to_csv(out_dir / "charging_stations.csv", index=False)

    pd.DataFrame({"Param": VEHICLE_PARAMS.keys(), "Value": VEHICLE_PARAMS.values()}).to_csv(out_dir / "vehicle_params.csv", index=False)

    print("Generated CSVs in", out_dir)


if __name__ == "__main__":
    main()
