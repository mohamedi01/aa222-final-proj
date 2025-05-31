import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data
stations_df = pd.read_csv("../data/charging_stations.csv")
route_df    = pd.read_csv("../data/route_segments.csv")

# Constants
B_MAX   = 80
B_MIN   = 10
B_START = 80

energy_per_mile = 0.25
total_energy_needed = route_df["Distance_mi"].sum() * energy_per_mile

num_stations = len(stations_df)
prices       = stations_df["Price_per_kWh"].values
max_charge_rate = stations_df["Max_Charge_Rate_kW"].values  # kW charging rates

# Objective weight for time penalty (you can tune this)
alpha = 0.1

def objective(q):
    # Linear cost part
    cost = np.dot(prices, q)
    # Time squared penalty part: sum ( (q_i / rate_i)^2 )
    #time_squared = np.sum((q / max_charge_rate) ** 2)
    return cost #+ alpha * time_squared

# Constraints
def soc_min(q):
    return B_START + q.sum() - total_energy_needed - B_MIN

def soc_cap(q):
    return B_MAX - (B_START + q.sum())

constraints = [
    {"type": "ineq", "fun": soc_min},
    {"type": "ineq", "fun": soc_cap},
]

bounds = [(0, 4*rate) for rate in max_charge_rate]  # max charging energy in 1 hour

q0 = np.zeros(num_stations)

res = minimize(
    objective,
    q0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"disp": True, "maxiter": 10000},
)

if res.success:
    print("\nOptimal charging plan (cost + time^2 objective):")
    for qi, (_, row) in zip(res.x, stations_df.iterrows()):
        print(f"{row['Station_Name']} (Segment {int(row['Segment_ID'])}): {qi:.2f} kWh")
    print(f"Total cost: ${np.dot(prices, res.x):.2f}")
    print(f"Time squared penalty: {np.sum((res.x / max_charge_rate) ** 2):.4f}")
    ls_cost = objective(res.x)
    lin_cost = np.dot(res.x, prices)
    batt_end = B_START + res.x.sum() - total_energy_needed
    print(f"\nLeast-squares objective value: {ls_cost:.4f}")
    print(f"Corresponding linear cost: ${lin_cost:.2f}")
    print(f"Battery at arrival: {batt_end:.2f} kWh (min {B_MIN} kWh)")

else:
    ls_cost = objective(res.x)
    lin_cost = np.dot(res.x, prices)
    batt_end = B_START + res.x.sum() - total_energy_needed
    print(f"\nLeast-squares objective value: {ls_cost:.4f}")
    print(f"Corresponding linear cost: ${lin_cost:.2f}")
    print(f"Battery at arrival: {batt_end:.2f} kWh (min {B_MIN} kWh)")
    raise RuntimeError("Optimization failed: " + res.message)

