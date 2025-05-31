#!/usr/bin/env python3
"""
create_trip.py  –  Generate mock CSVs for an EV road-trip
Stanford → USC, split into NUM_SEGMENTS, with realistic
segment distances and charging-station data.

Requirements (in your venv):
    pip install openrouteservice pandas numpy
"""

import os
import math
import openrouteservice
import pandas as pd
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────
ORS_API_KEY = os.environ.get("ORS_API_KEY", "5b3ce3597851110001cf62484616ffaedd464af7b1ebcf23f22eb681") 
NUM_SEGMENTS = 10                     # number of trip segments
MAX_WAYPOINTS_PER_CALL = 50           # keep < 70 → ORS safe
START_COORDS = (-122.1697, 37.4275)   # Stanford  (lon, lat)
END_COORDS   = (-118.2851, 34.0224)   # USC       (lon, lat)

VEHICLE_PARAMS = {
    "Initial_Battery_kWh": 75,
    "Battery_Capacity_kWh": 100,
    "Min_Battery_kWh": 10,
    "Efficiency_kWh_per_mi": 0.30,
    "Max_Speed_mph": 70,
    "Min_Speed_mph": 50,
}

# ── HELPERS ─────────────────────────────────────────────────────────────
def downsample(coords, max_points=MAX_WAYPOINTS_PER_CALL):
    """
    Keep first & last coord and subsample interior points so the list
    never exceeds `max_points`.  Ensures each ORS call stays < 70 pts.
    """
    n = len(coords)
    if n <= max_points:
        return coords

    step = math.ceil((n - 2) / (max_points - 2))  # reserve endpoints
    return [coords[0]] + coords[1:-1:step] + [coords[-1]]


# ── MAIN ────────────────────────────────────────────────────────────────
def main():
    client = openrouteservice.Client(key=ORS_API_KEY)

    # One full route request
    route = client.directions(
        coordinates=[START_COORDS, END_COORDS],
        profile="driving-car",
        format="geojson",
    )
    decoded_coords = route["features"][0]["geometry"]["coordinates"]
    total_pts = len(decoded_coords)
    step = total_pts // NUM_SEGMENTS

    # Build route_segments.csv
    segments = []
    for idx, i in enumerate(range(0, total_pts - step, step)):
        full_chunk = decoded_coords[i : i + step + 1]
        seg_coords = downsample(full_chunk)  # ≤ 50 waypoints

        seg_distance_m = client.directions(
            coordinates=seg_coords,
            profile="driving-car",
            format="geojson",
        )["features"][0]["properties"]["summary"]["distance"]

        segments.append(
            {
                "Segment_ID": idx,
                "Distance_mi": round(seg_distance_m / 1609.34, 2),  # m → mi
                "Elevation_Change_ft": int(np.random.randint(-300, 300)),
                "Avg_Temp_F": round(float(np.random.uniform(65, 80)), 1),
            }
        )

    pd.DataFrame(segments).to_csv("route_segments.csv", index=False)

    # Build charging_stations.csv (pick 4 random segments)
    station_segs = np.random.choice(
        [seg["Segment_ID"] for seg in segments], size=4, replace=False
    )
    stations = [
        {
            "Segment_ID": int(seg_id),
            "Station_Name": f"Station_{seg_id}",
            "Price_per_kWh": round(float(np.random.uniform(0.25, 0.45)), 2),
            "Max_Charge_Rate_kW": int(np.random.choice([50, 120, 150, 250])),
        }
        for seg_id in station_segs
    ]
    pd.DataFrame(stations).to_csv("charging_stations.csv", index=False)

    # Build vehicle_params.csv
    pd.DataFrame(
        {"Param": list(VEHICLE_PARAMS.keys()), "Value": list(VEHICLE_PARAMS.values())}
    ).to_csv("vehicle_params.csv", index=False)

    print(
        "✅  Generated: route_segments.csv, charging_stations.csv, vehicle_params.csv"
    )


if __name__ == "__main__":
    main()
