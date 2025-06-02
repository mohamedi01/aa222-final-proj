import pandas as pd
import numpy as np
import torch

# ────────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ────────────────────────────────────────────────────────────────────────────────
stations_df = pd.read_csv("../data/charging_stations.csv")
route_df    = pd.read_csv("../data/route_segments.csv")
route_df    = route_df.sort_values("Segment_ID").reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Constants & derived data
# ────────────────────────────────────────────────────────────────────────────────
B_MAX   = 80   # kWh
B_MIN   = 35   # kWh
B_START = 50   # kWh
energy_per_mile = 0.25
PENALTY = 1e5
MAX_TIME = 1

E_used_seg = route_df["Distance_mi"].values * energy_per_mile
n_segments = len(E_used_seg)

station_segments = stations_df["Segment_ID"].values.astype(int) - 1
num_stations     = len(stations_df)

prices      = stations_df["Price_per_kWh"].values
max_rate_kw = stations_df["Max_Charge_Rate_KW"].values if "Max_Charge_Rate_KW" in stations_df.columns else stations_df["Max_Charge_Rate_kW"].values

bounds = [(0.0, MAX_TIME * r) for r in max_rate_kw]
seg_to_station = {seg: idx for idx, seg in enumerate(station_segments)}

# ────────────────────────────────────────────────────────────────────────────────
# 3. PyTorch setup
# ────────────────────────────────────────────────────────────────────────────────
q = torch.zeros(num_stations, requires_grad=True)
optimizer = torch.optim.Adam([q], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)
num_iters = 1000

# Torch version of the cost function
def total_cost_torch(q):
    battery = B_START
    cost = torch.tensor(0.0)

    for seg_idx in range(n_segments):
        battery -= E_used_seg[seg_idx]

        if seg_idx in seg_to_station:
            st_idx = seg_to_station[seg_idx]
            charge = q[st_idx]
            battery += charge
            cost += charge * prices[st_idx]

        # Penalties
        if battery < B_MIN:
            cost += PENALTY * (B_MIN - battery)
        elif battery > B_MAX:
            cost += PENALTY * (battery - B_MAX)

    return cost

# ────────────────────────────────────────────────────────────────────────────────
# 4. Optimization loop
# ────────────────────────────────────────────────────────────────────────────────
for i in range(num_iters):
    optimizer.zero_grad()
    cost = total_cost_torch(q)
    cost.backward()
    optimizer.step()

    # Enforce bounds
    with torch.no_grad():
        for j, (lo, hi) in enumerate(bounds):
            q[j].clamp_(lo, hi)

    if i % 100 == 0 or i == num_iters - 1:
        print(f"Iter {i}: Cost = {cost.item():.2f}")

# ────────────────────────────────────────────────────────────────────────────────
# 5. Final results
# ────────────────────────────────────────────────────────────────────────────────
q_opt = q.detach().numpy()
print("\nOptimal charging plan:")
for qi, (_, row) in zip(q_opt, stations_df.iterrows()):
    print(f"{row['Station_Name']} (Segment {int(row['Segment_ID'])}): {qi:.2f} kWh")

print(f"\nTotal energy cost: ${np.dot(prices, q_opt):.2f}")

# SoC trace
battery = B_START
soc_trace = []
for seg_idx in range(n_segments):
    battery -= E_used_seg[seg_idx]
    if seg_idx in seg_to_station:
        battery += q_opt[seg_to_station[seg_idx]]
    soc_trace.append(battery)

print("SoC after each segment (kWh):", np.round(soc_trace, 2))
assert min(soc_trace) >= B_MIN - 1e-5, "SoC dipped below reserve!"
