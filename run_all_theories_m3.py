# run_all_theories_m3.py

import numpy as np
import time
from scipy.constants import G, c, k, hbar, epsilon_0
from gravity_compression_m3 import CompressionModel

DTYPE = np.float64
NUM_POINTS = 50_000

M_SOLAR = 1.989e30
M = np.array(M_SOLAR, dtype=DTYPE)
RS = np.array((2 * G * M) / (c**2), dtype=DTYPE)
J_PARAM = np.array(1e42, dtype=DTYPE)
Q_PARAM = np.array(1e12, dtype=DTYPE)
OBSERVER_ENERGY = np.array(1e9, dtype=DTYPE)
LAMBDA_COSMO = np.array(1.11e-52, dtype=DTYPE)

EPSILON = 1e-12
r = np.linspace(RS * 0.1, RS * 10, NUM_POINTS, dtype=DTYPE)
r[r == RS] += EPSILON

# --- All 17 Model Definitions ---
def compression_newtonian_limit(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    g_tt = -(1 - rs / r)
    g_rr = np.ones_like(r)
    return g_tt, g_rr

def compression_schwarzschild(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    denominator = 1 - rs / r
    g_tt = -denominator
    g_rr = 1 / (denominator + EPSILON)
    return g_tt, g_rr

def compression_reissner_nordstrom(r, M, Q=Q_PARAM, **kwargs):
    rs = (2 * G * M) / (c**2)
    r_Q_sq = (Q**2 * G) / (4 * np.pi * epsilon_0 * c**4)
    metric_term = 1 - rs / r + r_Q_sq / r**2
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_kerr_equatorial(r, M, J=J_PARAM, **kwargs):
    rs = (2 * G * M) / (c**2)
    a = J / (M * c)
    delta = r**2 - rs * r + a**2
    g_tt = -(delta - a**2) / r**2
    g_rr = r**2 / (delta + EPSILON)
    return g_tt, g_rr

def compression_yukawa_modified(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    lambda_range = 10 * rs
    yukawa_term = (rs / r) * np.exp(-r / lambda_range)
    metric_term = 1 - yukawa_term
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_quantum_corrected(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    alpha = 1.0
    quantum_correction = alpha * (rs**3 / r**3)
    metric_term = 1 - rs / r + quantum_correction
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_higher_dimensions(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    crossover_scale = 5 * rs
    four_d_term = rs / r
    five_d_term = (crossover_scale * rs) / r**2
    transition = 1 / (1 + np.exp(-(r - crossover_scale) / (crossover_scale/10)))
    metric_term = transition * four_d_term + (1 - transition) * five_d_term
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_log_corrected(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    beta = -0.5
    safe_r = np.maximum(r, rs * 1.001)
    log_correction = beta * (rs / safe_r) * np.log(safe_r / rs)
    metric_term = 1 - rs / r + log_correction
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_variable_g(r, M, **kwargs):
    rs_inf = (2 * G * M) / (c**2)
    delta = 0.1
    G_effective = G * (1 + delta * (rs_inf / r))
    rs_effective = (2 * G_effective * M) / (c**2)
    metric_term = rs_effective / r
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_non_local(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    cosmological_term = (LAMBDA_COSMO * r**2) / 3
    metric_term = 1 - rs / r - cosmological_term
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_fractal(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    D = 2.999
    base_potential = rs / r
    fractal_potential = base_potential**((D - 2.0))
    metric_term = fractal_potential
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_phase_transition(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    r_crit = 2 * rs
    normal_phase_term = 1 - rs / r
    condensed_phase_term = 1 - rs / r_crit
    metric_term = np.where(r > r_crit, normal_phase_term, condensed_phase_term)
    g_tt = -metric_term
    g_rr = 1 / (metric_term + EPSILON)
    return g_tt, g_rr

def compression_acausal(r, M, **kwargs):
    rs_initial = (2 * G * M) / (c**2)
    hawking_temp = (hbar * c**3) / (8 * np.pi * G * M * k)
    planck_temp = np.sqrt(hbar * c**5 / (G * k**2))
    correction_factor = 1 - (hawking_temp / planck_temp)
    rs_effective = rs_initial * correction_factor
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
    rs = (2 * G * M) / (c**2)
    fundamental_length = rs
    r_effective = r + fundamental_length**2 / r
    metric_term = rs / r_effective
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term + EPSILON)
    return g_tt, g_rr

def compression_hydrodynamic(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    v_flow_sq = (rs * c**2) / r
    safe_v_flow_sq = np.minimum(v_flow_sq, c**2 * 0.99999)
    gamma_sq = 1 / (1 - safe_v_flow_sq / c**2 + EPSILON)
    g_tt = -1 / gamma_sq
    g_rr = gamma_sq
    return g_tt, g_rr

def compression_participatory(r, M, observer_energy=OBSERVER_ENERGY, **kwargs):
    planck_energy = np.sqrt(hbar * c**5 / G)
    certainty = 1 - np.exp(-5 * observer_energy / planck_energy)
    rs = (2 * G * M) / (c**2)
    g_tt_gr, g_rr_gr = -(1 - rs / r), 1 / (1 - rs / r + EPSILON)
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
        ("Log Entropy Corrected",      {"func": compression_log_corrected, "params": {}}),
        ("Quantum Corrected",          {"func": compression_quantum_corrected, "params": {}}),
        ("Non-local (Cosmological)",   {"func": compression_non_local, "params": {}}),
        ("Emergent (Hydrodynamic)",    {"func": compression_hydrodynamic, "params": {}}),
        ("String T-Duality",           {"func": compression_t_duality, "params": {}}),
        ("Yukawa (Modified G)",        {"func": compression_yukawa_modified, "params": {}}),
        ("Higher-Dimensional",         {"func": compression_higher_dimensions, "params": {}}),
        ("Variable G",                 {"func": compression_variable_g, "params": {}}),
        ("Phase Transition",           {"func": compression_phase_transition, "params": {}}),
        ("Computational Complexity",   {"func": compression_computational, "params": {}}),
        ("Fractal Spacetime",          {"func": compression_fractal, "params": {}}),
        ("Participatory Universe",     {"func": compression_participatory, "params": {'observer_energy': OBSERVER_ENERGY}}),
        ("Newtonian Limit",            {"func": compression_newtonian_limit, "params": {}}),
    ]

    print("="*70)
    print("Apple M3 Max Analysis of Theoretical Models (Numerically Stable)")
    print(f"(Grid Resolution: {NUM_POINTS:,} points)")
    print("="*70)
    print(f"{'Rank':<5} | {'Model Name':<30} | {'Calculated Information Loss':<25}")
    print("-"*70)

    results = []
    for name, model_info in models_to_test:
        model = CompressionModel(compression_function=model_info["func"])
        loss = model.calculate_loss(r, M, **model_info["params"])
        results.append((name, loss))

    sorted_results = sorted(results, key=lambda item: item[1])

    for rank, (name, loss) in enumerate(sorted_results, 1):
        if np.isinf(loss):
            loss_str = "inf"
        elif np.isnan(loss):
            loss_str = "nan"
        else:
            loss_str = f"{loss:.6e}"
        print(f"{rank:<5} | {name:<30} | {loss_str:<25}")

    end_time = time.time()
    print("-"*70)
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print("="*70)