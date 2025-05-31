import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ---- LOAD DATA -------------------------------------------------------------
stations_df = pd.read_csv("../data/charging_stations.csv")
route_df    = pd.read_csv("../data/route_segments.csv")

# ---- VEHICLE CONSTANTS -----------------------------------------------------
B_MAX   = 80   # kWh capacity
B_MIN   = 10   # kWh required at arrival
B_START = 20   # kWh at departure

# For brevity we keep constant efficiency. Plug in your dynamic model if needed.
energy_per_mile = 0.25
total_energy_needed = route_df["Distance_mi"].sum() * energy_per_mile

# ---- DECISION VECTOR -------------------------------------------------------
num_stations = len(stations_df)
prices       = stations_df["Price_per_kWh"].values
max_charge   = stations_df["Max_Charge_Rate_kW"].values

# ---- LEAST‑SQUARES OBJECTIVE ----------------------------------------------
# Minimise sum_i ( sqrt(price_i) * q_i )^2  =  q^T diag(price) q
# which is a quadratic surrogate of the true linear cost.
def objective_ls(q):
    return np.dot(prices, q ** 2)  # Σ price_i * q_i^2

# ---- CONSTRAINTS -----------------------------------------------------------

def soc_min(q):
    """State of charge at arrival minus required minimum (≥ 0)."""
    return B_START + q.sum() - total_energy_needed - B_MIN

def soc_cap(q):
    """Remaining battery capacity at end of charging (≥ 0)."""
    return B_MAX - (B_START + q.sum())

constraints = [
    {"type": "ineq", "fun": soc_min},
    {"type": "ineq", "fun": soc_cap},
]

bounds = [(0, mc) for mc in max_charge]

# ---- SOLVE WITH SLSQP ------------------------------------------------------
q0 = np.zeros(num_stations)
res = minimize(
    objective_ls,
    q0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"disp": True, "maxiter": 10000},
)

if res.success:
    print("\nOptimal charging plan (least‑squares cost):")
    for qi, (_, row) in zip(res.x, stations_df.iterrows()):
        print(f"{row['Station_Name']} (Segment {int(row['Segment_ID'])}): {qi:.2f} kWh")
    ls_cost   = objective_ls(res.x)
    lin_cost  = np.dot(res.x, prices)
    batt_end  = B_START + res.x.sum() - total_energy_needed
    print(f"\nLeast‑squares objective value: {ls_cost:.4f}")
    print(f"Corresponding linear cost      : ${lin_cost:.2f}")
    print(f"Battery at arrival: {batt_end:.2f} kWh (min {B_MIN} kWh)")
else:
    ls_cost   = objective_ls(res.x)
    lin_cost  = np.dot(res.x, prices)
    batt_end  = B_START + res.x.sum() - total_energy_needed
    print(f"\nLeast‑squares objective value: {ls_cost:.4f}")
    print(f"Corresponding linear cost      : ${lin_cost:.2f}")
    print(f"Battery at arrival: {batt_end:.2f} kWh (min {B_MIN} kWh)")
    raise RuntimeError("Optimization failed: " + res.message)
