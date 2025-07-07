# run_all_theories_m3.py

# Import NumPy as the primary array library for Apple Silicon.
import numpy as np
# Import the time library to measure execution speed.
import time
# Import physical constants for accurate calculations.
from scipy.constants import G, c, k_b, hbar, Lambda, epsilon_0
# Import the M3-optimized model from its separate file.
from gravity_compression_m3 import CompressionModel

# --- HPC OPTIMIZATION: High-density grid for precision runs ---
DTYPE = np.float32 # Use 32-bit floats for better performance.
NUM_POINTS = 1_000_000 # A high number of points for a precision run.

# --- Global Simulation Parameters (Initialized as NumPy arrays) ---
M_SOLAR = 1.989e30
M = np.array(M_SOLAR, dtype=DTYPE)
RS = np.array((2 * G * M) / (c**2), dtype=DTYPE)
J_PARAM = np.array(1e42, dtype=DTYPE)
Q_PARAM = np.array(1e12, dtype=DTYPE)
OBSERVER_ENERGY = np.array(1e9, dtype=DTYPE)
r = np.linspace(RS * 0.1, RS * 10, NUM_POINTS, dtype=DTYPE)

# --- All 17 Model Definitions ---

# 1. Newtonian: Models gravity affecting time but not space.
def compression_newtonian_limit(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    g_tt = -(1 - rs / r)
    g_rr = np.ones_like(r)
    return g_tt, g_rr

# 2. Schwarzschild: The correct GR solution for a simple mass.
def compression_schwarzschild(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    g_tt = -(1 - rs / r)
    g_rr = 1 / (1 - rs / r)
    return g_tt, g_rr

# 3. Reissner-Nordström: Includes the effect of electric charge.
def compression_reissner_nordstrom(r, M, Q=Q_PARAM, **kwargs):
    rs = (2 * G * M) / (c**2)
    r_Q_sq = (Q**2 * G) / (4 * np.pi * epsilon_0 * c**4)
    metric_term = 1 - rs / r + r_Q_sq / r**2
    g_tt = -metric_term
    g_rr = 1 / metric_term
    return g_tt, g_rr

# 4. Kerr: Includes the effect of rotation (frame-dragging).
def compression_kerr_equatorial(r, M, J=J_PARAM, **kwargs):
    rs = (2 * G * M) / (c**2)
    a = J / (M * c)
    delta = r**2 - rs * r + a**2
    g_tt = -(delta - a**2) / r**2
    g_rr = r**2 / delta
    return g_tt, g_rr

# 5. Yukawa: Models gravity as a short-range force.
def compression_yukawa_modified(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    lambda_range = 10 * rs
    yukawa_term = (rs / r) * np.exp(-r / lambda_range)
    g_tt = -(1 - yukawa_term)
    g_rr = 1 / (1 - yukawa_term)
    return g_tt, g_rr

# 6. Quantum Corrected: Adds a term to prevent singularities.
def compression_quantum_corrected(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    alpha = 1.0
    quantum_correction = alpha * (rs**3 / r**3)
    metric_term = 1 - rs / r + quantum_correction
    g_tt = -metric_term
    g_rr = 1 / metric_term
    return g_tt, g_rr

# 7. Higher-Dimensional: Models gravity leaking into extra dimensions.
def compression_higher_dimensions(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    crossover_scale = 5 * rs
    four_d_term = rs / r
    five_d_term = (crossover_scale * rs) / r**2
    transition = 1 / (1 + np.exp(-(r - crossover_scale) / (crossover_scale/10)))
    metric_term = transition * four_d_term + (1 - transition) * five_d_term
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term)
    return g_tt, g_rr

# 8. Logarithmic Corrected: Adds a quantum entropy correction to geometry.
def compression_log_corrected(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    beta = -0.5
    safe_r = np.maximum(r, rs * 1.001)
    log_correction = beta * (rs / safe_r) * np.log(safe_r / rs)
    metric_term = 1 - rs / r + log_correction
    g_tt = -metric_term
    g_rr = 1 / metric_term
    return g_tt, g_rr

# 9. Variable G: Models the gravitational constant as a field.
def compression_variable_g(r, M, **kwargs):
    rs_inf = (2 * G * M) / (c**2)
    delta = 0.1
    G_effective = G * (1 + delta * (rs_inf / r))
    rs_effective = (2 * G_effective * M) / (c**2)
    metric_term = rs_effective / r
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term)
    return g_tt, g_rr

# 10. Non-local: Includes the effect of the universe's expansion.
def compression_non_local(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    cosmological_term = (Lambda * r**2) / 3
    metric_term = 1 - rs / r - cosmological_term
    g_tt = -metric_term
    g_rr = 1 / metric_term
    return g_tt, g_rr

# 11. Fractal Spacetime: Models a non-integer spatial dimension.
def compression_fractal(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    D = 2.999
    base_potential = rs / r
    fractal_potential = base_potential**((D - 2.0))
    metric_term = fractal_potential
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term)
    return g_tt, g_rr

# 12. Phase Transition: Models spacetime changing properties near the source.
def compression_phase_transition(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    r_crit = 2 * rs
    normal_phase_term = 1 - rs / r
    condensed_phase_term = 1 - rs / r_crit
    metric_term = np.where(r > r_crit, normal_phase_term, condensed_phase_term)
    g_tt = -metric_term
    g_rr = 1 / metric_term
    return g_tt, g_rr

# 13. Acausal: Links the geometry to its final thermodynamic state.
def compression_acausal(r, M, **kwargs):
    rs_initial = (2 * G * M) / (c**2)
    hawking_temp = (hbar * c**3) / (8 * np.pi * G * M * k_b)
    planck_temp = np.sqrt(hbar * c**5 / (G * k_b**2))
    correction_factor = 1 - (hawking_temp / planck_temp)
    rs_effective = rs_initial * correction_factor
    g_tt = -(1 - rs_effective / r)
    g_rr = 1 / (1 - rs_effective / r)
    return g_tt, g_rr

# 14. Computational Complexity: Relates curvature to information density.
def compression_computational(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    planck_length = np.sqrt(G * hbar / c**3)
    safe_r = np.maximum(r, planck_length)
    complexity_factor = (rs**2)
    metric_term = complexity_factor / (safe_r * np.log2(safe_r / planck_length))
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term)
    return g_tt, g_rr

# 15. T-Duality: Models a String Theory symmetry between large and small scales.
def compression_t_duality(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    fundamental_length = rs
    r_effective = r + fundamental_length**2 / r
    metric_term = rs / r_effective
    g_tt = -(1 - metric_term)
    g_rr = 1 / (1 - metric_term)
    return g_tt, g_rr

# 16. Hydrodynamic: Models gravity as an emergent fluid-like effect.
def compression_hydrodynamic(r, M, **kwargs):
    rs = (2 * G * M) / (c**2)
    v_flow_sq = (rs * c**2) / r
    safe_v_flow_sq = np.minimum(v_flow_sq, c**2 * 0.99999)
    gamma_sq = 1 / (1 - safe_v_flow_sq / c**2)
    g_tt = -1 / gamma_sq
    g_rr = gamma_sq
    return g_tt, g_rr

# 17. Participatory: Models geometry as actualized by observation.
def compression_participatory(r, M, observer_energy=OBSERVER_ENERGY, **kwargs):
    planck_energy = np.sqrt(hbar * c**5 / G)
    certainty = 1 - np.exp(-5 * observer_energy / planck_energy)
    rs = (2 * G * M) / (c**2)
    g_tt_gr, g_rr_gr = -(1 - rs / r), 1 / (1 - rs / r)
    g_tt_vac, g_rr_vac = np.full_like(r, -1.0), np.ones_like(r)
    g_tt = certainty * g_tt_gr + (1 - certainty) * g_tt_vac
    g_rr = certainty * g_rr_gr + (1 - certainty) * g_rr_vac
    return g_tt, g_rr


# This block ensures the code below only runs when the script is executed directly.
if __name__ == "__main__":
    start_time = time.time()

    # The re-ordered list of theories to test.
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
    
    # Print a header for the results table.
    print("="*70)
    print("Apple M3 Max Analysis of Theoretical Models")
    print(f"(Grid Resolution: {NUM_POINTS:,} points)")
    print("="*70)
    print(f"{'Rank':<5} | {'Model Name':<30} | {'Calculated Information Loss':<25}")
    print("-"*70)

    results = []
    # Loop through the re-ordered list of models to test.
    for name, model_info in models_to_test:
        model = CompressionModel(compression_function=model_info["func"])
        loss = model.calculate_information_loss(r, M, **model_info["params"])
        results.append((name, loss))

    # Sort results by loss value to confirm the ranking.
    sorted_results = sorted(results, key=lambda item: item[1])

    # Print the final, ranked results table.
    for rank, (name, loss) in enumerate(sorted_results, 1):
        print(f"{rank:<5} | {name:<30} | {loss:.6e}")

    end_time = time.time()
    # Print a final summary of the execution.
    print("-"*70)
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print("="*70)
