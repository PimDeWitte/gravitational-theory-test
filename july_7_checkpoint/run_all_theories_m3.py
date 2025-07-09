# run_geodesic_test.py

import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.constants import G, c, k, hbar, epsilon_0
from gravity_compression_m3 import GravitationalTheory, geodesic_ode

# --- Simulation Parameters ---
DTYPE = np.float64
M = 1.989e30 * 10  # Mass of a 10-solar-mass black hole
RS = (2 * G * M) / (c**2)
EPSILON = 1e-12
J_PARAM = 1e42
Q_PARAM = 1e12
OBSERVER_ENERGY = 1e9
LAMBDA_COSMO = 1.11e-52

print(f"Maximum Precision Geodesic Test for a {M/1.989e30:.1f} M_sol Black Hole (Rs = {RS:.2f} m)\n")

# --- All Theory Implementations as Subclasses ---

class Schwarzschild(GravitationalTheory):
    def __init__(self): super().__init__("Schwarzschild (GR)")
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        denominator = 1 - rs_local / r
        return denominator, 1 / (denominator + EPSILON)

class NewtonianLimit(GravitationalTheory):
    def __init__(self): super().__init__("Newtonian Limit")
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        return 1 - rs_local / r, 1.0

class ReissnerNordstrom(GravitationalTheory):
    def __init__(self, Q):
        super().__init__(f"Reissner-Nordström")
        self.Q = Q
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        r_Q_sq = (self.Q**2 * G) / (4 * np.pi * epsilon_0 * C**4)
        metric_term = 1 - rs_local / r + r_Q_sq / r**2
        return -metric_term, 1 / (metric_term + EPSILON)

class Kerr(GravitationalTheory):
    def __init__(self, J, M, C):
        super().__init__(f"Kerr (Equatorial)")
        self.a = J / (M * C)
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        delta = r**2 - rs_local * r + self.a**2
        return -(delta - self.a**2) / r**2, r**2 / (delta + EPSILON)

class Yukawa(GravitationalTheory):
    def __init__(self, lambda_mult):
        super().__init__(f"Yukawa (λ={lambda_mult:.2f}*RS)")
        self.lambda_mult = lambda_mult
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        lambda_range = self.lambda_mult * rs_local
        yukawa_term = (rs_local / r) * np.exp(-r / lambda_range)
        metric_term = 1 - yukawa_term
        return -metric_term, 1 / (1 - metric_term + EPSILON)

class QuantumCorrected(GravitationalTheory):
    def __init__(self, alpha):
        super().__init__(f"Quantum Corrected (α={alpha:.2f})")
        self.alpha = alpha
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        quantum_correction = self.alpha * (rs_local**3 / r**3)
        metric_term = 1 - rs_local / r + quantum_correction
        return -metric_term, 1 / (metric_term + EPSILON)

class HigherDimensional(GravitationalTheory):
    def __init__(self, crossover_mult):
        super().__init__(f"Higher-Dim (cross={crossover_mult:.2f}*RS)")
        self.crossover_mult = crossover_mult
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        crossover_scale = self.crossover_mult * rs_local
        four_d_term = rs_local / r
        five_d_term = (crossover_scale * rs_local) / r**2
        transition = 1 / (1 + np.exp(-(r - crossover_scale) / (crossover_scale/10)))
        metric_term = transition * four_d_term + (1 - transition) * five_d_term
        return -(1 - metric_term), 1 / (1 - metric_term + EPSILON)

class LogCorrected(GravitationalTheory):
    def __init__(self, beta):
        super().__init__(f"Log Corrected (β={beta:.2f})")
        self.beta = beta
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        safe_r = np.maximum(r, rs_local * 1.001)
        log_correction = self.beta * (rs_local / safe_r) * np.log(safe_r / rs_local)
        metric_term = 1 - rs_local / r + log_correction
        return -metric_term, 1 / (metric_term + EPSILON)

class VariableG(GravitationalTheory):
    def __init__(self, delta):
        super().__init__(f"Variable G (δ={delta:.2f})")
        self.delta = delta
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        G_effective = G * (1 + self.delta * (rs_local / r))
        rs_effective = (2 * G_effective * M) / (C**2)
        metric_term = rs_effective / r
        return -(1 - metric_term), 1 / (1 - metric_term + EPSILON)

class NonLocal(GravitationalTheory):
    def __init__(self): super().__init__("Non-local (Cosmological)")
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        cosmological_term = (LAMBDA_COSMO * r**2) / 3
        metric_term = 1 - rs_local / r - cosmological_term
        return -metric_term, 1 / (metric_term + EPSILON)

class Fractal(GravitationalTheory):
    def __init__(self, D):
        super().__init__(f"Fractal (D={D:.2f})")
        self.D = D
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        base_potential = rs_local / r
        fractal_potential = base_potential**((self.D - 2.0))
        metric_term = fractal_potential
        return -(1 - metric_term), 1 / (1 - metric_term + EPSILON)

class PhaseTransition(GravitationalTheory):
    def __init__(self, crit_mult):
        super().__init__(f"Phase Transition (crit={crit_mult:.2f}*RS)")
        self.crit_mult = crit_mult
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        r_crit = self.crit_mult * rs_local
        normal_phase_term = 1 - rs_local / r
        condensed_phase_term = 1 - rs_local / r_crit
        metric_term = np.where(r > r_crit, normal_phase_term, condensed_phase_term)
        return -metric_term, 1 / (metric_term + EPSILON)

class Acausal(GravitationalTheory):
    def __init__(self): super().__init__("Acausal (Final State)")
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        hawking_temp = (hbar * C**3) / (8 * np.pi * G * M * k)
        planck_temp = np.sqrt(hbar * C**5 / (G * k**2))
        correction_factor = 1 - (hawking_temp / planck_temp)
        rs_effective = rs_local * correction_factor
        metric_term = 1 - rs_effective / r
        return -metric_term, 1 / (metric_term + EPSILON)

class Computational(GravitationalTheory):
    def __init__(self): super().__init__("Computational Complexity")
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        planck_length = np.sqrt(G * hbar / C**3)
        safe_r = np.maximum(r, planck_length)
        complexity_factor = (rs_local**2)
        metric_term = complexity_factor / (safe_r * np.log2(safe_r / planck_length))
        return -(1 - metric_term), 1 / (1 - metric_term + EPSILON)

class Tduality(GravitationalTheory):
    def __init__(self): super().__init__("String T-Duality")
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        fundamental_length = rs_local
        r_effective = r + fundamental_length**2 / r
        metric_term = rs_local / r_effective
        return -(1 - metric_term), 1 / (1 - metric_term + EPSILON)

class Hydrodynamic(GravitationalTheory):
    def __init__(self): super().__init__("Emergent (Hydrodynamic)")
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        v_flow_sq = (rs_local * C**2) / r
        safe_v_flow_sq = np.minimum(v_flow_sq, C**2 * 0.99999)
        gamma_sq = 1 / (1 - safe_v_flow_sq / C**2 + EPSILON)
        return -1 / gamma_sq, gamma_sq

class Participatory(GravitationalTheory):
    def __init__(self, observer_energy):
        super().__init__(f"Participatory (E_obs={observer_energy:.1e})")
        self.observer_energy = observer_energy
    def get_metric(self, r, M, C, G, **kwargs):
        rs_local = (2 * G * M) / (C**2)
        planck_energy = np.sqrt(hbar * C**5 / G)
        certainty = 1 - np.exp(-5 * self.observer_energy / planck_energy)
        g_tt_gr, g_rr_gr = -(1 - rs_local / r), 1 / (1 - rs_local / r + EPSILON)
        g_tt_vac, g_rr_vac = np.full_like(r, -1.0), np.ones_like(r)
        g_tt = certainty * g_tt_gr + (1 - certainty) * g_tt_vac
        g_rr = certainty * g_rr_gr + (1 - certainty) * g_rr_vac
        return g_tt, g_rr


if __name__ == "__main__":
    start_time = time.time()

    models_to_test = [
        Schwarzschild(),
        NewtonianLimit(),
        Acausal(),
        Kerr(J=J_PARAM, M=M, C=c),
        ReissnerNordstrom(Q=Q_PARAM),
        NonLocal(),
        Computational(),
        Tduality(),
        Hydrodynamic(),
        Participatory(observer_energy=OBSERVER_ENERGY)
    ]
    
    param_sweeps = {
        "Quantum Corrected": {"class": QuantumCorrected, "params": {"alpha": np.linspace(-2.0, 2.0, 10)}},
        "Log Corrected": {"class": LogCorrected, "params": {"beta": np.linspace(-1.5, 1.5, 10)}},
        "Yukawa": {"class": Yukawa, "params": {"lambda_mult": np.logspace(np.log10(1.5), 2, 10)}},
        "Variable G": {"class": VariableG, "params": {"delta": np.linspace(-0.5, 0.5, 10)}},
        "Fractal Spacetime": {"class": Fractal, "params": {"D": np.linspace(2.95, 3.05, 10)}},
        "Phase Transition": {"class": PhaseTransition, "params": {"crit_mult": [1.5, 2.5, 4.0, 8.0, 16.0]}},
        "Higher-Dimensional": {"class": HigherDimensional, "params": {"crossover_mult": [2.0, 10.0, 20.0, 50.0]}}
    }
    
    for name, config in param_sweeps.items():
        model_class = config["class"]
        for param_name, param_values in config["params"].items():
            for val in param_values:
                models_to_test.append(model_class(**{param_name: val}))

    total_models = len(models_to_test)
    print("="*80)
    print("Apple M3 Max Maximum Precision Geodesic Analysis")
    print(f"(Total Models: {total_models})")
    print("="*80)
    
    # --- PRECISION INCREASE 1: Longer infall path ---
    r0 = 100 * RS
    g_tt_initial, _ = models_to_test[0].get_metric(r0, M, c, G)
    dt_dtau_initial = 1.0 / np.sqrt(g_tt_initial)
    y0 = [0.0, r0, 0.0, dt_dtau_initial, 0.0, 0.0]
    
    # --- PRECISION INCREASE 2: Longer simulation time ---
    tau_span = [0, 5_000_000]
    tau_eval = np.array([tau_span[1]])
    
    # --- PRECISION INCREASE 3: Tighter solver tolerances ---
    TOLERANCE = 1e-13

    print("Running ground truth simulation with Schwarzschild (GR)...")
    sol_true = solve_ivp(
        geodesic_ode, tau_span, y0, args=(models_to_test[0], M, c, G), 
        method='RK45', t_eval=tau_eval, rtol=TOLERANCE, atol=TOLERANCE
    )
    ground_truth_final_state = sol_true.y[:, -1]
    r_true, phi_true = ground_truth_final_state[1], ground_truth_final_state[2]
    print(f"Ground truth final position: r={r_true/RS:.2f}*Rs, φ={np.rad2deg(phi_true):.2f}°\n")

    results = []
    print("--- Testing Candidate Models by Trajectory ---")
    for model in models_to_test:
        print(f"Testing: {model.name}...")
        start_time_model = time.time()
        
        sol_pred = solve_ivp(
            geodesic_ode, tau_span, y0, args=(model, M, c, G), 
            method='RK45', t_eval=tau_eval, rtol=TOLERANCE, atol=TOLERANCE
        )
        predicted_final_state = sol_pred.y[:, -1]
        r_pred, phi_pred = predicted_final_state[1], predicted_final_state[2]
        
        loss = (r_true**2 + r_pred**2 - 2 * r_true * r_pred * np.cos(phi_true - phi_pred))

        results.append({"Model": model.name, "Loss": loss})
        print(f"  -> Finished in {time.time() - start_time_model:.2f}s. Loss: {loss:.4e}")

    print("\n--- High-Precision Trajectory Test: Ranked Performance ---")
    results.sort(key=lambda x: x["Loss"])
    print(f"{'Rank':<5} | {'Model Name':<50} | {'Trajectory Loss (m^2)':<25}")
    print("-"*80)
    for rank, res in enumerate(results, 1):
        print(f"{rank:<5} | {res['Model']:<50} | {res['Loss']:.4e}")
    
    print("-"*80)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("="*80)