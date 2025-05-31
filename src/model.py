import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load your CSV files
stations_df = pd.read_csv("charging_stations.csv")
route_df = pd.read_csv("route_segments.csv")

# Constants
B_max = 80       # max battery kWh
B_min = 10       # min battery kWh at end
B_start = 20     # starting battery kWh

# Calculate total energy needed for route (approximate)
energy_per_mile = 0.25  # kWh per mile
total_energy_needed = np.sum(route_df['Distance_mi']) * energy_per_mile  # kWh

# Decision variables: amount charged at each station (kWh)
num_stations = len(stations_df)
prices = stations_df['Price_per_kWh'].values
max_charge = stations_df['Max_Charge_Rate_kW'].values  # max charge per station (assuming max 1 hour charge)

# Objective function with penalty for constraints
def objective(x):
    # Cost = sum of (energy charged * price)
    cost = np.dot(x, prices)
    
    penalty = 0
    total_charge = np.sum(x)
    battery_end = B_start + total_charge - total_energy_needed
    
    # Penalty for violating minimum battery left at end of trip
    if battery_end < B_min:
        penalty += 1e6 * (B_min - battery_end)**2
    
    # Penalty for exceeding max battery capacity
    if (B_start + total_charge) > B_max:
        penalty += 1e6 * ((B_start + total_charge) - B_max)**2
    
    # Penalty for charging more than max charging rate at any station
    for i in range(num_stations):
        if x[i] < 0:
            penalty += 1e6 * (abs(x[i]))**2
        if x[i] > max_charge[i]:
            penalty += 1e6 * (x[i] - max_charge[i])**2
    
    return cost + penalty

# Initial guess: zero charge at all stations
x0 = np.zeros(num_stations)

# Run Nelder-Mead
result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter':10000, 'disp': True})

if result.success:
    print("Optimal charging amounts at stations:")
    for i, station in stations_df.iterrows():
        print(f"{station['Station_Name']} (Segment {station['Segment_ID']}): {result.x[i]:.2f} kWh")
    total_cost = np.dot(result.x, prices)
    print(f"Total charging cost: ${total_cost:.2f}")
else:
    print("Optimization failed:", result.message)
