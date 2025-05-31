#!/usr/bin/env python3
"""optimize.py  –  Linear-program solution for EV charging cost minimization

Usage (defaults assume CSVs from create_trip.py are in cwd):
    python optimize.py
    # or explicitly
    python optimize.py --route route_segments.csv --stations charging_stations.csv

Requires: numpy, pandas, scipy (HiGHS backend)
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ── VEHICLE & MODEL CONSTANTS ────────────────────────────
ENERGY_PER_MILE = 0.25  # kWh consumed per mile
B_MAX = 80              # Battery capacity (kWh)
B_MIN = 10              # Minimum battery allowed at arrival (kWh)
B_START = 20            # Battery level at departure (kWh)


# ── LP BUILD HELPERS ─────────────────────────────────────
def build_problem(route_csv: str, stations_csv: str):
    """Read CSVs and build LP matrices for linprog."""
    stations_df = pd.read_csv(stations_csv)
    route_df = pd.read_csv(route_csv)

    num_stations = len(stations_df)
    prices = stations_df["Price_per_kWh"].to_numpy()
    max_charge = stations_df["Max_Charge_Rate_kW"].to_numpy()

    # Total energy the vehicle must supply over the trip
    energy_needed = route_df["Distance_mi"].sum() * ENERGY_PER_MILE

    # Objective vector (cost per kWh at each station)
    c = prices.copy()

    # Inequality constraints  A_ub x ≤ b_ub
    # 1) End-of-trip SoC ≤ B_MAX  ⟹  Σq_i ≤ B_MAX − B_START
    # 2) End-of-trip SoC ≥ B_MIN  ⟹  −Σq_i ≤ −(B_MIN + energy_needed − B_START)
    A_ub = np.vstack([ np.ones(num_stations), -np.ones(num_stations) ])
    b_ub = np.array([ B_MAX - B_START,
                     -(B_MIN + energy_needed - B_START) ])

    # Bounds: 0 ≤ q_i ≤ station max
    bounds = [(0, mc) for mc in max_charge]

    return stations_df, c, A_ub, b_ub, bounds, energy_needed


def solve_lp(c, A_ub, b_ub, bounds):
    """Run linprog with HiGHS."""
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)
    return res


def print_plan(res, stations_df: pd.DataFrame, energy_needed: float):
    """Pretty-print charging amounts, total cost, and end SoC."""
    total_cost = 0.0
    print("\nOptimal charging plan\n" + "-" * 28)
    for q, (_, row) in zip(res.x, stations_df.iterrows()):
        print(f"{row['Station_Name']} (Segment {int(row['Segment_ID'])}): {q:.2f} kWh")
        total_cost += q * row['Price_per_kWh']
    batt_end = B_START + res.x.sum() - energy_needed
    print(f"\nTotal charging cost: ${total_cost:.2f}")
    print(f"Battery at arrival: {batt_end:.2f} kWh  (min allowed {B_MIN} kWh)")


# ── MAIN ENTRY ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optimize EV charging plan along a route.")
    parser.add_argument("--route", default="route_segments.csv", help="CSV file of route segments")
    parser.add_argument("--stations", default="charging_stations.csv", help="CSV file of charging stations")
    args = parser.parse_args()

    stations_df, c, A_ub, b_ub, bounds, energy_needed = build_problem(args.route, args.stations)
    result = solve_lp(c, A_ub, b_ub, bounds)
    print_plan(result, stations_df, energy_needed)


if __name__ == "__main__":
    main()