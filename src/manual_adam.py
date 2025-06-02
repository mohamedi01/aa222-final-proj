import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Problem Setup
# --------------------------
num_stations = 5
ev_capacity_kwh = 100
initial_charge_kwh = 20
target_charge_kwh = 80
distance_between_stations_km = 100
km_per_kwh = 5
energy_needed_per_leg = distance_between_stations_km / km_per_kwh  # 20 kWh per leg

max_rate_kw = 50  # Max charging rate at any station
time_between_stations = 1  # 1 hour between stations
total_legs = num_stations - 1
T = num_stations  # number of decision points

# Electricity prices at each station
prices = np.array([0.25, 0.30, 0.20, 0.35, 0.28])

# Bounds for charging rates (q_t)
bounds = [(0, max_rate_kw) for _ in range(T)]

# --------------------------
# Charging Simulation
# --------------------------
def charging_route(q_values):
    soc = [initial_charge_kwh]
    charge = initial_charge_kwh
    for i, q in enumerate(q_values):
        charge += q * time_between_stations
        if i < total_legs:
            charge -= energy_needed_per_leg
        soc.append(charge)
    return np.array(soc)

# --------------------------
# Objective Function
# --------------------------
def total_cost(q_values):
    soc = charging_route(q_values)
    penalty = 0
    for i, s in enumerate(soc):
        if s < 0 or s > ev_capacity_kwh:
            penalty += 1000 + 1000 * abs(s - ev_capacity_kwh if s > ev_capacity_kwh else s)
    if soc[-1] < target_charge_kwh:
        penalty += 1000 * (target_charge_kwh - soc[-1])
    return np.dot(q_values, prices) + penalty

# --------------------------
# Hooke-Jeeves Optimizer
# --------------------------
def hooke_jeeves(f, x0, bounds, step_size=1.0, step_decay=0.5, tol=1e-3, max_iter=200):
    x = x0.copy()
    n = len(x)
    steps = np.full(n, step_size)
    cost_trace = []
    x_trace = []

    def explore(base, steps):
        x_new = base.copy()
        for i in range(n):
            for direction in [+1, -1]:
                trial = x_new.copy()
                trial[i] += direction * steps[i]
                trial[i] = np.clip(trial[i], bounds[i][0], bounds[i][1])
                if f(trial) < f(x_new):
                    x_new = trial
                    break
        return x_new

    for k in range(max_iter):
        x_new = explore(x, steps)
        if np.allclose(x, x_new, atol=tol):
            steps *= step_decay
            if np.all(steps < tol):
                print(f"Stopping at iteration {k} due to small step size.")
                break
        else:
            direction = x_new - x
            x = x_new
            x_trial = x_new + direction
            x_trial = np.clip(x_trial, [b[0] for b in bounds], [b[1] for b in bounds])
            if f(x_trial) < f(x_new):
                x = x_trial
        cost = f(x)
        cost_trace.append(cost)
        x_trace.append(x.copy())
        print(f"Iter {k}: Cost = {cost:.2f}")
    return x, x_trace, cost_trace

# --------------------------
# Run Optimization
# --------------------------
np.random.seed(80)
q_init = np.random.uniform(low=0.0, high=max_rate_kw, size=T)
q_opt, q_trace, cost_trace = hooke_jeeves(total_cost, q_init, bounds, step_size=5.0)

# --------------------------
# Visualization
# --------------------------
soc = charging_route(q_opt)
stations = np.arange(num_stations)

plt.figure(figsize=(12, 5))

# SoC Plot
plt.subplot(1, 2, 1)
plt.plot(stations, soc, marker='o')
plt.title("Battery SoC at Each Station")
plt.xlabel("Station")
plt.ylabel("SoC (kWh)")
plt.ylim(0, ev_capacity_kwh + 10)
plt.grid(True)

# Cost Trace
plt.subplot(1, 2, 2)
plt.plot(cost_trace)
plt.title("Total Cost vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Total Cost ($)")
plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------
# Print Results
# --------------------------
print("\nCharging Profile:")
for i, q in enumerate(q_opt):
    print(f"  Station {i}: Charge for {q:.2f} kWh")

print(f"\nFinal SoC: {soc[-1]:.2f} kWh (Target: {target_charge_kwh} kWh)")
print(f"Total Cost: ${total_cost(q_opt):.2f}")
