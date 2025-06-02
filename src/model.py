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
from scipy.optimize import minimize, basinhopping

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
RANDOM_SEARCH_TRIALS = 100
RANDOM_SEARCH = False

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

bounds = [(0.0, r) for r in max_rate_kw]

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


def q0(max_rate_kw):
    np.random.seed(80)
    upper_bounds = np.array(max_rate_kw)
    q = np.random.uniform(low=0.0, high=upper_bounds)
    return q
print("STATRTING POINT", q0(max_rate_kw))

q_trace = []
cost_trace = []

def track_q(q):
    cost = total_cost(q)
    q_trace.append(q.copy())
    cost_trace.append(cost)
    print(f"Iteration {len(q_trace)}: cost = {cost:.2f}, q = {q}")
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
            if RANDOM_SEARCH: break
            cost += PENALTY * (B_MIN - battery)
        elif battery > B_MAX:
            if RANDOM_SEARCH: break
            cost += PENALTY * (battery - B_MAX)

    cost += TIME_PENALTY * (np.sum(q / max_rate_kw)) ** 2
    cost_trace.append(cost)
    q_trace.append(q.copy())
    return cost

# rng = np.random.default_rng(99)
# best_rand_cost = np.inf
# best_rand_q    = None
# RANDOM_SEARCH = True
# for _ in range(RANDOM_SEARCH_TRIALS):
#     q_rand = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
#     c_rand = total_cost(q_rand)
#     if c_rand < best_rand_cost:
#         best_rand_cost, best_rand_q = c_rand, q_rand
# RANDOM_SEARCH = False
# print(f"Random search best cost over {RANDOM_SEARCH_TRIALS} trials: {best_rand_cost:.2f}")
# total kWh the wheels will consume
#-------------------------------------------------------------------------------------------------------
# choosing each charger in each segment
energy_need = E_used_seg.sum()
# net kWh that must be added through charging to finish at B_MIN
required_kWh = max(0.0, energy_need + B_MIN - B_START)

q_fixed = np.zeros(num_stations)
capacities = np.array([b[1] for b in bounds])

# total kWh the wheels will consume
energy_need = E_used_seg.sum()
# net kWh that must be added through charging to finish at B_MIN
required_kWh = max(0.0, energy_need + B_MIN - B_START)

q_fixed = np.zeros(num_stations)
capacities = np.array([b[1] for b in bounds])

if required_kWh > 0:
    # initial even allocation
    per_station = required_kWh / num_stations
    q_fixed = np.minimum(per_station, capacities)
    remaining = required_kWh - q_fixed.sum()

    # distribute any leftover to stations that still have headroom
    if remaining > 1e-9:
        headroom = capacities - q_fixed
        while remaining > 1e-9 and headroom.sum() > 0:
            share = remaining / (headroom > 0).sum()
            add = np.minimum(share, headroom)
            q_fixed += add
            remaining -= add.sum()
            headroom = capacities - q_fixed

fixed_cost = total_cost(q_fixed)
print(f"Even‑distribution heuristic cost: {fixed_cost:.2f}")

# SoC trace under even‑distribution plan
battery_fixed = B_START
soc_fixed_trace = []
for seg_idx in range(n_segments):
    battery_fixed -= E_used_seg[seg_idx]
    for st_idx in seg_to_stations.get(seg_idx, []):
        battery_fixed += q_fixed[st_idx]
    soc_fixed_trace.append(battery_fixed)
print("Heuristic SoC after each segment:", np.round(soc_fixed_trace, 2))

# choosing the cheapest in a segment
chosen_idx = []
for seg in range(n_segments):
    idxs = seg_to_stations.get(seg, [])
    if idxs:                     # segment has at least one charger
        cheapest = idxs[np.argmin(prices[idxs])]
        chosen_idx.append(cheapest)

if not chosen_idx:
    raise RuntimeError("No stations selected by heuristic")

chosen_idx = np.array(chosen_idx)
n_chosen   = len(chosen_idx)

# 2) total kWh we must add so arrival SoC == B_MIN
required_kWh = max(0.0, B_MIN + E_used_seg.sum() - B_START)

# 3) give every chosen station the same amount, capped by its max-rate bound
capacities   = np.array([bounds[i][1] for i in chosen_idx])
q_equal_seg  = np.zeros(num_stations)

if required_kWh > 0:
    per = required_kWh / n_chosen
    per = min(per, capacities.min())        # respect tightest cap
    q_equal_seg[chosen_idx] = per           # identical charge

# 4) evaluate and report
equal_cost = total_cost(q_equal_seg)
print(f"\nOne-station-per-segment equal-share cost: {equal_cost:.2f}")

# SoC trace for this heuristic
soc_equal = []
bat = B_START
for s in range(n_segments):
    bat -= E_used_seg[s]
    for i in seg_to_stations.get(s, []):
        bat += q_equal_seg[i]
    soc_equal.append(bat)
print("Heuristic SoC after each segment:", np.round(soc_equal, 2))


# stopping only when about to run out of charge and 
# charging a constant amount
cheapest_idx = {seg: idxs[np.argmin(prices[idxs])]
                for seg, idxs in seg_to_stations.items()}

def simulate(dose):
    """
    Return (cost_no_penalty, final_soc, per-seg SoC list, q_vec) given a fixed charge `dose`.
    We compute cost manually (no penalty), because we ensure SoC never dips below B_MIN.
    If at any point soc < B_MIN and there is no station to charge, we mark infeasible by returning np.inf cost.
    """
    q_vec = np.zeros(num_stations)
    soc   = B_START
    soc_path = []
    cost = 0.0

    for seg in range(n_segments):
        # Forecast post-drive SoC
        future_soc = soc - E_used_seg[seg]

        # If that would dip below B_MIN, charge first (if possible)
        if future_soc < B_MIN:
            if seg not in cheapest_idx:
                # No station this segment → infeasible
                return np.inf, future_soc, soc_path, q_vec

            idx   = cheapest_idx[seg]
            added = min(dose, bounds[idx][1])  # respect cap
            q_vec[idx] += added
            cost      += added * prices[idx]
            soc       += added

            # Recompute future_soc after topping up
            future_soc = soc - E_used_seg[seg]

        # Now apply the drive step
        soc = future_soc

        # Record SoC after this segment
        soc_path.append(soc)

    # After loop, soc ≥ B_MIN by construction
    # Add the time penalty term (no other penalties)
    cost += (np.sum(q_vec / max_rate_kw)) ** 2

    return cost, soc, soc_path, q_vec

# --------------- Bisection to find the best dose that ends SoC near B_MIN ---------------
lo, hi = 0.0, max_rate_kw.min()
best_q      = None
best_diff   = np.inf
best_soc_end = None
best_soc_path = None

for _ in range(25):
    mid = 0.5 * (lo + hi)
    cost_mid, soc_end, soc_mid_path, q_mid = simulate(mid)

    if cost_mid == np.inf:
        # infeasible dose → SoC dipped below B_MIN at some segment
        lo = mid
        continue

    # track how close soc_end is to B_MIN
    diff = abs(soc_end - B_MIN)
    if diff < best_diff:
        best_diff    = diff
        best_q       = q_mid.copy()
        best_soc_end = soc_end
        best_soc_path = soc_mid_path.copy()

    if soc_end > B_MIN:
        # still overshooting → try smaller dose
        hi = mid
    else:
        # undershooting → try larger dose
        lo = mid

if best_q is not None:
    cost_fix = np.dot(best_q, prices) + (np.sum(best_q / max_rate_kw)) ** 2
    print(f"\nFixed-dose heuristic cost: {cost_fix:.2f}  (final SoC = {best_soc_end:.2f} kWh)")
    print("Heuristic SoC after each segment:\n", np.round(best_soc_path, 2))
else:
    print("\nFixed-dose heuristic: no feasible constant dose found.")

#-------------------------------------------------------------------------------------------------------
minimizer_kwargs = { 
    "method": "SLSQP",
    "bounds": bounds,
    "options": {"disp": False, "maxiter": 100, "ftol":1e-5}}

res = basinhopping(
    total_cost,
    q0(max_rate_kw),
    minimizer_kwargs=minimizer_kwargs,
    niter=5,
    stepsize=0.1,
    seed=80
)
# ────────────────────────────────────────────────────────────────────────────────
# 7.  Optimisation (SLSQP)
# ────────────────────────────────────────────────────────────────────────────────
# res = minimize(total_cost, q0(max_rate_kw),#np.zeros(num_stations), 
#                method="SLSQP", bounds=bounds, callback=track_q, options={"disp": True, "maxiter": 50, "ftol":1e-5})
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

# plt.figure(figsize=(12, 12))
# # --- Plot objective function over iterations ---
# # plt.figure(figsize=(10, 4))
# plt.subplot(2, 1, 1)
# plt.plot(cost_trace, label="Objective Function")
# plt.xlabel("Iteration")
# plt.ylabel("Total Cost ($)")
# plt.title("Objective Function Value vs. Iteration")
# plt.grid(True)
# plt.legend()
# # plt.tight_layout()
# # plt.show()

# # --- Plot charge amount per station vs. iteration ---
# q_trace_arr = np.array(q_trace)  # shape: (iterations, num_stations)
# # plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 2)
# for i in range(q_trace_arr.shape[1]):
#     plt.plot(q_trace_arr[:, i], label=f"Station {i+1}")
# plt.xlabel("Iteration")
# plt.ylabel("Charge Amount (kWh)")
# plt.title("Charge Decisions vs. Iteration")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# #Log
# plt.figure(figsize=(10, 4))
# plt.plot(np.log(cost_trace), label="Objective Function")
# plt.xlabel("Iteration")
# plt.ylabel("Total Cost ($)")
# plt.title("Objective Function Value vs. Iteration")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()