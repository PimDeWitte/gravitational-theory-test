# run_all_theories_HPC.py

import numpy as np
import time
from scipy.constants import G, c, k, hbar, epsilon_0
from gravity_compression_m3 import CompressionModel

# --- HPC OPTIMIZATION: High-density grid for maximum precision ---
DTYPE = np.float64
NUM_POINTS = 50_000_000

# --- Global Simulation Parameters ---
M_SOLAR = 1.989e30
M = np.array(M_SOLAR, dtype=DTYPE)
RS = np.array((2 * G * M) / (c**2), dtype=DTYPE)
J_PARAM = np.array(1e42, dtype=DTYPE)
Q_PARAM = np.array(1e12, dtype=DTYPE)
OBSERVER_ENERGY = np.array(1e9, dtype=DTYPE)
LAMBDA_COSMO = np.array(1.11e-52, dtype=DTYPE)

EPSILON = 1e-15
r = np.linspace(RS * 0.1, RS * 10, NUM_POINTS, dtype=DTYPE)
r[r == RS] += EPSILON

# --- Base Model Definitions (17 Theories) ---

def compression_newtonian_limit(r, M, **kwargs):
    g_tt = -(1 - RS / r)
    g_rr = np.ones_like(r)
    return g_tt, g_rr

def compression_schwarzschild(r, M, **kwargs):
    denominator = 1 - RS / r
    g_tt = -denominator
    g_rr = 1 / (denominator + EPSILON)
    return g_tt, g_rr

def compression_reissner_nordstrom(r, M, Q=Q_PARAM, **kwargs):
    r_Q_sq = (Q**2 * G) / (4 * np.pi * epsilon_0 * c**4)
    metric_term = 1 - RS / r + r_Q_sq / r**2
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_kerr_equatorial(r, M, J=J_PARAM, **kwargs):
    a = J / (M * c)
    delta = r**2 - RS * r + a**2
    g_tt = -(delta - a**2) / r**2
    g_rr = r**2 / (delta + EPSILON)
    return g_tt, g_rr

def compression_yukawa_modified(r, M, lambda_mult=10.0, **kwargs):
    lambda_range = lambda_mult * RS
    yukawa_term = (RS / r) * np.exp(-r / lambda_range)
    metric_term = 1 - yukawa_term
    g_tt = -metric_term
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_quantum_corrected(r, M, alpha=1.0, **kwargs):
    quantum_correction = alpha * (RS**3 / r**3)
    metric_term = 1 - RS / r + quantum_correction
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_higher_dimensions(r, M, crossover_mult=5.0, **kwargs):
    crossover_scale = crossover_mult * RS
    four_d_term = RS / r
    five_d_term = (crossover_scale * RS) / r**2
    transition = 1 / (1 + np.exp(-(r - crossover_scale) / (crossover_scale/10)))
    metric_term = transition * four_d_term + (1 - transition) * five_d_term
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_log_corrected(r, M, beta=-0.5, **kwargs):
    safe_r = np.maximum(r, RS * 1.001)
    log_correction = beta * (RS / safe_r) * np.log(safe_r / RS)
    metric_term = 1 - RS / r + log_correction
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_variable_g(r, M, delta=0.1, **kwargs):
    G_effective = G * (1 + delta * (RS / r))
    rs_effective = (2 * G_effective * M) / (c**2)
    metric_term = rs_effective / r
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_non_local(r, M, **kwargs):
    cosmological_term = (LAMBDA_COSMO * r**2) / 3
    metric_term = 1 - RS / r - cosmological_term
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_fractal(r, M, D=2.999, **kwargs):
    base_potential = RS / r
    fractal_potential = base_potential**((D - 2.0))
    metric_term = fractal_potential
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_phase_transition(r, M, crit_mult=2.0, **kwargs):
    r_crit = crit_mult * RS
    normal_phase_term = 1 - RS / r
    condensed_phase_term = 1 - RS / r_crit
    metric_term = np.where(r > r_crit, normal_phase_term, condensed_phase_term)
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_acausal(r, M, **kwargs):
    hawking_temp = (hbar * c**3) / (8 * np.pi * G * M * k)
    planck_temp = np.sqrt(hbar * c**5 / (G * k**2))
    correction_factor = 1 - (hawking_temp / planck_temp)
    rs_effective = RS * correction_factor
    metric_term = 1 - rs_effective / r
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_computational(r, M, **kwargs):
    planck_length = np.sqrt(G * hbar / c**3)
    safe_r = np.maximum(r, planck_length)
    complexity_factor = (RS**2)
    metric_term = complexity_factor / (safe_r * np.log2(safe_r / planck_length))
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_t_duality(r, M, **kwargs):
    fundamental_length = RS
    r_effective = r + fundamental_length**2 / r
    metric_term = RS / r_effective
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_hydrodynamic(r, M, **kwargs):
    v_flow_sq = (RS * c**2) / r
    safe_v_flow_sq = np.minimum(v_flow_sq, c**2 * 0.99999)
    gamma_sq = 1 / (1 - safe_v_flow_sq / c**2 + EPSILON)
    g_tt = -1 / gamma_sq
    g_rr = gamma_sq
    return g_tt, g_rr

def compression_participatory(r, M, observer_energy=OBSERVER_ENERGY, **kwargs):
    planck_energy = np.sqrt(hbar * c**5 / G)
    certainty = 1 - np.exp(-5 * observer_energy / planck_energy)
    g_tt_gr, g_rr_gr = -(1 - RS / r), 1 / (1 - RS / r + EPSILON)
    g_tt_vac, g_rr_vac = np.full_like(r, -1.0), np.ones_like(r)
    g_tt = certainty * g_tt_gr + (1 - certainty) * g_tt_vac
    g_rr = certainty * g_rr_gr + (1 - certainty) * g_rr_vac
    return g_tt, g_rr

if __name__ == "__main__":
    start_time = time.time()

    models_to_test = [
        ("Schwarzschild (GR)",         {"func": compression_schwarzschild, "params": {}}),
        ("Acausal (Final State)",      {"func": compression_acausal, "params": {}}),
        ("Kerr (Equatorial)",          {"func": compression_kerr_equatorial, "params": {'J': J_PARAM}}),
        ("Reissner-Nordström",         {"func": compression_reissner_nordstrom, "params": {'Q': Q_PARAM}}),
        ("Non-local (Cosmological)",   {"func": compression_non_local, "params": {}}),
        ("Emergent (Hydrodynamic)",    {"func": compression_hydrodynamic, "params": {}}),
        ("String T-Duality",           {"func": compression_t_duality, "params": {}}),
        ("Computational Complexity",   {"func": compression_computational, "params": {}}),
        ("Participatory Universe",     {"func": compression_participatory, "params": {'observer_energy': OBSERVER_ENERGY}}),
        ("Newtonian Limit",            {"func": compression_newtonian_limit, "params": {}}),
    ]
    
    param_sweeps = {
        "Quantum Corrected": {"func": compression_quantum_corrected, "params": {"alpha": np.linspace(-2.0, 2.0, 10)}},
        "Log Corrected": {"func": compression_log_corrected, "params": {"beta": np.linspace(-1.5, 1.5, 10)}},
        "Yukawa": {"func": compression_yukawa_modified, "params": {"lambda_mult": np.logspace(np.log10(1.5), 2, 10)}},
        "Variable G": {"func": compression_variable_g, "params": {"delta": np.linspace(-0.5, 0.5, 10)}},
        "Fractal Spacetime": {"func": compression_fractal, "params": {"D": np.linspace(2.95, 3.05, 10)}},
        "Phase Transition": {"func": compression_phase_transition, "params": {"crit_mult": [1.5, 2.5, 4.0, 8.0, 16.0]}},
        "Higher-Dimensional": {"func": compression_higher_dimensions, "params": {"crossover_mult": [2.0, 10.0, 20.0, 50.0]}}
    }
    
    for name, config in param_sweeps.items():
        func = config["func"]
        for param_name, param_values in config["params"].items():
            for val in param_values:
                test_name = f"{name} ({param_name}={val:.2f})"
                models_to_test.append((test_name, {"func": func, "params": {param_name: val}}))

    total_models = len(models_to_test)
    print("="*80)
    print("Apple M3 Max High-Precision Analysis of Theoretical Models (Massive Scale)")
    print(f"(Grid Resolution: {NUM_POINTS:,} points | Total Models: {total_models})")
    print("="*80)
    print(f"{'Rank':<5} | {'Model Name':<50} | {'Calculated Information Loss':<25}")
    print("-"*80)

    results = []
    for name, model_info in models_to_test:
        model = CompressionModel(compression_function=model_info["func"])
        loss = model.calculate_loss(r, M, **model_info["params"])
        results.append((name, loss))

    sorted_results = sorted(results, key=lambda item: item[1])

    for rank, (name, loss) in enumerate(sorted_results, 1):
        if np.isinf(loss): loss_str = "inf"
        elif np.isnan(loss): loss_str = "nan"
        else: loss_str = f"{loss:.6e}"
        print(f"{rank:<5} | {name:<50} | {loss_str:<25}")

    end_time = time.time()
    print("-"*80)
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print("="*80)