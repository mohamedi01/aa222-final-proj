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
B_MIN   = 35   # kWh reserve
B_START = 50   # kWh initial

energy_per_mile = 0.25

E_used_seg = route_df["Distance_mi"].values * energy_per_mile
n_segments = len(E_used_seg)

station_segments = stations_df["Segment_ID"].values.astype(int) - 1
num_stations     = len(stations_df)

prices      = stations_df["Price_per_kWh"].values
max_rate_kw = stations_df["Max_Charge_Rate_KW"].values if "Max_Charge_Rate_KW" in stations_df.columns else stations_df["Max_Charge_Rate_kW"].values  # handle column naming

bounds = [(0.0, 4.0 * r) for r in max_rate_kw]  # 4‑hour cap per stop

# Quick feasibility: even if we charge to the upper bound everywhere, does the
# battery remain above B_MIN?
max_possible_soc = B_START + np.cumsum([bounds[seg_to_station[s]][1] if (s := i) in (seg_to_station := {seg: idx for idx, seg in enumerate(station_segments)}) else 0 for i in range(n_segments)]) - np.cumsum(E_used_seg)

if max_possible_soc.min() < B_MIN:
    raise RuntimeError(
        "Infeasible route: even with maximum allowable energy the battery dips below reserve."
    )

# ────────────────────────────────────────────────────────────────────────────────
# 3.  Objective with penalty AFTER charging
# ────────────────────────────────────────────────────────────────────────────────
PENALTY = 1e5  # big enough to dominate $/kWh ≈ 0.30

seg_to_station = {seg: idx for idx, seg in enumerate(station_segments)}

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
    options={"disp": True, "maxiter": 20000},
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

print("SoC after each segment (kWh):", np.round(soc_trace, 2))
assert min(soc_trace) >= B_MIN - 1e-5, "SoC dipped below reserve!"
