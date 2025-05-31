"""EV Route Charging Optimizer — objective‑only variant (penalty after charging)

Fixes why the optimiser never purchased energy:

* Penalty for dropping below reserve is now evaluated **after** the optional
  charging stop at each segment.
* Adds a quick feasibility check; if the maximum allowable energy still leaves
  the battery below the reserve the script warns and exits.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Load data
# ────────────────────────────────────────────────────────────────────────────────
stations_df = pd.read_csv("../data/charging_stations.csv")
route_df    = pd.read_csv("../data/route_segments.csv")

route_df = route_df.sort_values("Segment_ID").reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# 2.  Constants & derived data
# ────────────────────────────────────────────────────────────────────────────────
B_MAX   = 80   # kWh pack size
B_MIN   = 1   # kWh reserve
B_START = 80   # kWh initial
PENALTY = 1e5              # $ penalty factor for SoC violations
TIME_PENALTY = 1000

# --- Base consumption and adjustment factors -----------------------------------
BASE_KWH_PER_MILE = 0.25   # flat road, 60 mph, 70 °F, no HVAC

TEMP_REF_F   = 70          # reference temperature (°F)
TEMP_SLOPE   = 0.006       # +0.6 % energy per |Δ°F|

SPEED_REF_MPH = 60         # aerodynamic drag reference (mph)
SPEED_QUAD    = 0.002      # +0.2 % per (Δ mph)^2

ELEV_UP_COEFF = 0.0003       # kWh per ft climbed per mile (rough regen‑ignored)
ELEV_REGEN_EFF = 0.60      # 60 % of uphill cost comes back on descent
# Regenerative braking ---------------------------------------------
TRAFFIC_SEED = 27                              # reproducible randomness
REGEN_KWH_PER_MILE_AT_FULL_STOPGO = 0.05       # kWh/mi recovered at traffic=1

if "Traffic_Intensity" in route_df.columns:
    traffic_profile = route_df["Traffic_Intensity"].clip(0, 1).values.astype(float)
else:
    rng = np.random.default_rng(TRAFFIC_SEED)
    traffic_profile = rng.uniform(0.1, 0.7, size=len(route_df))  # moderate traffic range
    route_df["Traffic_Intensity"] = traffic_profile  # keep for inspection

def energy_per_mile(row) -> float:
    """Return kWh/mile adjusted for temp, speed, elevation, *and regen*."""
    # Temperature adjustment (HVAC + chemistry)
    temp_factor = TEMP_SLOPE * abs(row["Avg_Temp_F"] - TEMP_REF_F)

    # Speed adjustment (aero)
    speed_factor = SPEED_QUAD * (row["Speed_limit"] - SPEED_REF_MPH) ** 2

    # Elevation cost (climb and fall to model energy spent and regen)
    elev_change_ft = row["Elevation_Change_ft"]      # can be + or –
    if elev_change_ft >= 0:
        elev_factor = elev_change_ft * ELEV_UP_COEFF
    else:
        elev_factor = elev_change_ft * ELEV_UP_COEFF * ELEV_REGEN_EFF
    # print(elev_factor)

    # Regenerative braking benefit (subtractive term)
    traffic_intensity = row["Traffic_Intensity"]
    regen_benefit = REGEN_KWH_PER_MILE_AT_FULL_STOPGO * traffic_intensity

    # Ensure we never go below some tiny positive value
    return max(0, BASE_KWH_PER_MILE * (1 + temp_factor + speed_factor) + elev_factor - regen_benefit)

# Compute per‑segment energy usage vector ---------------------------------------
E_used_seg = route_df.apply(lambda r: energy_per_mile(r) * r["Distance_mi"], axis=1).values
n_segments = len(E_used_seg)

station_segments = stations_df["Segment_ID"].values.astype(int) - 1
num_stations     = len(stations_df)

prices      = stations_df["Price_per_kWh"].values
max_rate_kw = (stations_df["Max_Charge_Rate_KW"] if "Max_Charge_Rate_KW" in stations_df.columns else stations_df["Max_Charge_Rate_kW"]).values

bounds = [(0.0, 4.0 * r) for r in max_rate_kw]  # kWh, assuming ≤4 h stop
seg_to_station = {seg: idx for idx, seg in enumerate(station_segments)}

# Quick feasibility: even if we charge to the upper bound everywhere, does the
# battery remain above B_MIN?
upper_energy_by_seg = np.zeros(n_segments)
for seg_idx, qbound in zip(station_segments, bounds):
    upper_energy_by_seg[seg_idx] = qbound[1]

max_possible_soc = B_START + np.cumsum(upper_energy_by_seg) - np.cumsum(E_used_seg)
if max_possible_soc.min() < B_MIN:
    raise RuntimeError("Route infeasible: even max charging cannot keep SoC above reserve.")

# ────────────────────────────────────────────────────────────────────────────────
# 3.  Objective with penalty AFTER charging
# ────────────────────────────────────────────────────────────────────────────────
def total_cost(q: np.ndarray) -> float:
    battery = B_START
    cost    = 0.0

    for seg_idx in range(n_segments):
        # Drive the segment
        battery -= E_used_seg[seg_idx]

        # Optional charge *at this same segment*
        if seg_idx in seg_to_station:
            st_idx = seg_to_station[seg_idx]
            charge = q[st_idx]
            battery += charge
            cost   += charge * prices[st_idx]

        # Penalties after both drive and charge
        if battery < B_MIN:
            cost += PENALTY * (B_MIN - battery)
        elif battery > B_MAX:
            cost += PENALTY * (battery - B_MAX)

    charging_time = np.sum(q / max_rate_kw)

    # Add squared charging time penalty
    cost += TIME_PENALTY * charging_time**2

    return cost

# ────────────────────────────────────────────────────────────────────────────────
# 4.  Optimisation
# ────────────────────────────────────────────────────────────────────────────────
init_q = np.zeros(num_stations)
res = minimize(
    total_cost,
    init_q,
    method="SLSQP",
    bounds=bounds,
    options={"disp": True, "maxiter": 500},
)

if not res.success:
    raise RuntimeError("Optimization failed: " + res.message)

q_opt = res.x

# ────────────────────────────────────────────────────────────────────────────────
# 5.  Results
# ────────────────────────────────────────────────────────────────────────────────
print("\nOptimal charging plan:")
for qi, (_, row) in zip(q_opt, stations_df.iterrows()):
    print(f"{row['Station_Name']} (Segment {int(row['Segment_ID'])}): {qi:.2f} kWh")

print(f"\nTotal energy cost: ${np.dot(prices, q_opt):.2f}")

battery = B_START
soc_trace = []
for seg_idx in range(n_segments):
    battery -= E_used_seg[seg_idx]
    if seg_idx in seg_to_station:
        battery += q_opt[seg_to_station[seg_idx]]
    soc_trace.append(battery)

energy_per_mile_constants = route_df.apply(lambda r: energy_per_mile(r), axis=1).values

print(seg_to_station)
print("SoC after each segment (kWh):", np.round(soc_trace, 2))
print("Energy per segment:", [f'{e:.2f}' for e in E_used_seg])
print("Energy per mile for each segment:", [f'{e:.2f}' for e in energy_per_mile_constants])
assert min(soc_trace) >= B_MIN - 1e-5, "SoC dipped below reserve!"
