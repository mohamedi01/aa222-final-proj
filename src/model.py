"""EV Route Charging Optimizer — allow multiple stations per segment

Same as the previous version *but* restores the richer diagnostics block that
prints:
  * the segment→stations mapping (`seg_to_stations`),
  * state‑of‑charge trace,
  * energy used per segment, and
  * per‑segment energy‑per‑mile constants.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

cost_trace = []
q_trace = []

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Load data
# ────────────────────────────────────────────────────────────────────────────────
stations_df = pd.read_csv("../data/charging_stations.csv")
route_df    = pd.read_csv("../data/route_segments.csv").sort_values("Segment_ID").reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# 2.  Constants & coefficients
# ────────────────────────────────────────────────────────────────────────────────
B_MAX, B_MIN, B_START = 80.0, 10.0, 30.0
PENALTY       = 1e5      # SoC violation
TIME_PENALTY  = 1000    # weight for squared charging time

BASE_KWH_PER_MILE = 0.25
TEMP_REF_F, TEMP_SLOPE = 70, 0.006
SPEED_REF_MPH, SPEED_QUAD = 60, 0.002
ELEV_UP_COEFF, ELEV_REGEN_EFF = 0.0003, 0.60
TRAFFIC_SEED = 27
REGEN_KWH_PER_MILE_AT_FULL_STOPGO = 0.05

# ────────────────────────────────────────────────────────────────────────────────
# 3.  Traffic profile & per‑segment consumption
# ────────────────────────────────────────────────────────────────────────────────
if "Traffic_Intensity" in route_df:
    traffic_profile = route_df["Traffic_Intensity"].clip(0, 1).values.astype(float)
else:
    rng = np.random.default_rng(TRAFFIC_SEED)
    traffic_profile = rng.uniform(0.1, 0.7, size=len(route_df))
    route_df["Traffic_Intensity"] = traffic_profile


def energy_per_mile(row):
    temp_factor  = TEMP_SLOPE * abs(row["Avg_Temp_F"] - TEMP_REF_F)
    speed_factor = SPEED_QUAD * (row["Speed_limit"] - SPEED_REF_MPH) ** 2
    elev         = row["Elevation_Change_ft"]
    elev_factor  = elev * ELEV_UP_COEFF * (1 if elev >= 0 else ELEV_REGEN_EFF)
    regen        = REGEN_KWH_PER_MILE_AT_FULL_STOPGO * row["Traffic_Intensity"]
    return max(0, BASE_KWH_PER_MILE * (1 + temp_factor + speed_factor) + elev_factor - regen)

# Per‑segment energy‑per‑mile constants and energy usage
energy_per_mile_constants = route_df.apply(energy_per_mile, axis=1).values
E_used_seg = energy_per_mile_constants * route_df["Distance_mi"].values
n_segments = len(E_used_seg)

# ────────────────────────────────────────────────────────────────────────────────
# 4.  Station data structures (multiple per segment allowed)
# ────────────────────────────────────────────────────────────────────────────────
station_segments = stations_df["Segment_ID"].values.astype(int) - 1
num_stations     = len(stations_df)

prices      = stations_df["Price_per_kWh"].values
max_rate_kw = stations_df.get("Max_Charge_Rate_KW", stations_df.get("Max_Charge_Rate_kW")).values

bounds = [(0.0, 4.0 * r) for r in max_rate_kw]

seg_to_stations = {}
for st_idx, seg_idx in enumerate(station_segments):
    if seg_idx not in seg_to_stations:
        seg_to_stations[seg_idx] = []
    seg_to_stations[seg_idx].append(st_idx)

# ────────────────────────────────────────────────────────────────────────────────
# 5.  Feasibility check
# ────────────────────────────────────────────────────────────────────────────────
upper_energy_by_seg = np.zeros(n_segments)
for st_idx, seg_idx in enumerate(station_segments):
    upper_energy_by_seg[seg_idx] += bounds[st_idx][1]

if (B_START + np.cumsum(upper_energy_by_seg) - np.cumsum(E_used_seg)).min() < B_MIN:
    raise RuntimeError("Route infeasible: even max charging cannot keep SoC above reserve.")

# ────────────────────────────────────────────────────────────────────────────────
# 6.  Objective (continuous)
# ────────────────────────────────────────────────────────────────────────────────

def total_cost(q: np.ndarray) -> float:
    battery = B_START
    cost    = 0.0

    for seg_idx in range(n_segments):
        battery -= E_used_seg[seg_idx]
        for st_idx in seg_to_stations.get(seg_idx, []):
            charge = q[st_idx]
            battery += charge
            cost    += charge * prices[st_idx]

        if battery < B_MIN:
            cost += PENALTY * (B_MIN - battery)
        elif battery > B_MAX:
            cost += PENALTY * (battery - B_MAX)

    cost += TIME_PENALTY * (np.sum(q / max_rate_kw)) ** 2
    cost_trace.append(cost)
    q_trace.append(q.copy())
    return cost

# ────────────────────────────────────────────────────────────────────────────────
# 7.  Optimisation (SLSQP)
# ────────────────────────────────────────────────────────────────────────────────
res = minimize(total_cost, np.zeros(num_stations), 
               method="SLSQP", bounds=bounds, options={"disp": True, "maxiter": 50, "ftol":1e-5})
if not res.success:
    raise RuntimeError("Optimization failed: " + res.message)
q_opt = res.x

# ────────────────────────────────────────────────────────────────────────────────
# 8.  Results & diagnostics
# ────────────────────────────────────────────────────────────────────────────────
print("\nOptimal charging plan (multiple chargers allowed per segment):")
for qi, (_, row) in zip(q_opt, stations_df.iterrows()):
    print(f"{row['Station_Name']} (Seg {int(row['Segment_ID'])}): {qi:.2f} kWh")

print(f"\nTotal energy cost: ${np.dot(prices, q_opt):.2f}")

battery, soc_trace = B_START, []
for seg_idx in range(n_segments):
    battery -= E_used_seg[seg_idx]
    for st_idx in seg_to_stations.get(seg_idx, []):
        battery += q_opt[st_idx]
    soc_trace.append(battery)

print("\nseg_to_stations mapping:", dict(seg_to_stations))
print("SoC after each segment (kWh):", np.round(soc_trace, 2))
print("Energy per segment (kWh):", [f"{e:.2f}" for e in E_used_seg])
print("Energy per mile for each segment (kWh/mi):", [f"{e:.3f}" for e in energy_per_mile_constants])
assert min(soc_trace) >= B_MIN - 1e-5, "SoC dipped below reserve!"

import matplotlib.pyplot as plt
print(len(q_trace))
print(len(cost_trace))

plt.figure(figsize=(12, 12))
# --- Plot objective function over iterations ---
# plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(cost_trace, label="Objective Function")
plt.xlabel("Iteration")
plt.ylabel("Total Cost ($)")
plt.title("Objective Function Value vs. Iteration")
plt.grid(True)
plt.legend()
# plt.tight_layout()
# plt.show()

# --- Plot charge amount per station vs. iteration ---
q_trace_arr = np.array(q_trace)  # shape: (iterations, num_stations)
# plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 2)
for i in range(q_trace_arr.shape[1]):
    plt.plot(q_trace_arr[:, i], label=f"Station {i+1}")
plt.xlabel("Iteration")
plt.ylabel("Charge Amount (kWh)")
plt.title("Charge Decisions vs. Iteration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Log
plt.figure(figsize=(10, 4))
plt.plot(np.log(cost_trace), label="Objective Function")
plt.xlabel("Iteration")
plt.ylabel("Total Cost ($)")
plt.title("Objective Function Value vs. Iteration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
