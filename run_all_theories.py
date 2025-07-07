# run_all_theories.py (Optimized)

import numpy as np
import time # Import time to measure execution speed
from scipy.constants import G, c, k_b, hbar, Lambda, epsilon_0
from gravity_compression import CompressionModel

# --- Global Simulation Parameters ---
M_SOLAR = 1.989e30
M = M_SOLAR

# --- OPTIMIZATION 1: Pre-calculate all constants ---
# These values don't change, so we compute them only once.
RS = (2 * G * M) / (c**2)
J_PARAM = 1e42
Q_PARAM = 1e12
PLANCK_ENERGY = np.sqrt(hbar * c**5 / G)
OBSERVER_ENERGY = 1e9 # Using Planck Energy as a default for the participatory model

# --- OPTIMIZATION 2: Reduce grid resolution ---
# The number of points is the biggest factor in performance.
# 500 points is often sufficient for a smooth plot and is much faster than 2000.
NUM_POINTS = 500
r = np.linspace(RS * 0.1, RS * 10, NUM_POINTS)

# --- Definitions of All 17 Suggested Models (Optimized) ---
# Functions are modified to accept 'rs' directly, avoiding recalculation.

def compression_newtonian_limit(r, M, rs, **kwargs):
    g_tt = -(1 - rs / r)
    g_rr = np.ones_like(r)
    return g_tt, g_rr

def compression_schwarzschild(r, M, rs, **kwargs):
    g_tt = -(1 - rs / r)
    g_rr = 1 / (1 - rs / r)
    return g_tt, g_rr

def compression_reissner_nordstrom(r, M, rs, Q, **kwargs):
    r_Q_sq = (Q**2 * G) / (4 * np.pi * epsilon_0 * c**4)
    metric_term = 1 - rs / r + r_Q_sq / r**2
    g_tt = -metric_term
    g_rr = 1 / metric_term
    return g_tt, g_rr

def compression_kerr_equatorial(r, M, rs, J, **kwargs):
    a = J / (M * c)
    delta = r**2 - rs * r + a**2
    g_tt = -(delta - a**2) / r**2
    g_rr = r**2 / delta
    return g_tt, g_rr

# (All other 13 functions would be similarly modified to accept 'rs' if they use it)
# For brevity, only the first few are shown modified. The logic remains the same.
# ... (include all other 13 functions from the previous response here) ...

if __name__ == "__main__":
    start_time = time.time() # Start timer

    models_to_test = {
        "Newtonian Limit":            {"func": compression_newtonian_limit, "params": {'rs': RS}},
        "Schwarzschild (GR)":         {"func": compression_schwarzschild, "params": {'rs': RS}},
        "Reissner-Nordström":         {"func": compression_reissner_nordstrom, "params": {'rs': RS, 'Q': Q_PARAM}},
        "Kerr (Equatorial)":          {"func": compression_kerr_equatorial, "params": {'rs': RS, 'J': J_PARAM}},
        # ... (all other models would be added here with their parameters) ...
    }

    print("="*60)
    print("Running All Theoretical Models Against Information Loss Metric")
    print("="*60)
    print(f"{'Model Name':<30} | {'Calculated Information Loss':<25}")
    print("-"*60)

    results = {}
    for name, model_info in models_to_test.items():
        model = CompressionModel(compression_function=model_info["func"])
        # --- OPTIMIZATION 3: Pass all params in a single dictionary ---
        # The model's 'calculate_information_loss' now takes M and the pre-calculated params.
        loss = model.calculate_information_loss(r, M, **model_info["params"])
        results[name] = loss

    sorted_results = sorted(results.items(), key=lambda item: item[1])

    for name, loss in sorted_results:
        print(f"{name:<30} | {loss:.4e}")

    end_time = time.time() # End timer
    print("-"*60)
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print("="*60)
