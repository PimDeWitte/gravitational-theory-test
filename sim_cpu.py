# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# A. LIBRARY IMPORTS AND SETUP
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.constants import G, c, k, hbar, epsilon_0
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# B. SIMULATION PARAMETERS (High-Precision float64)
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# Set the default data type to float64 for maximum precision. NumPy uses this by default on 64-bit systems.
DTYPE = np.float64
# Mass of the central object (10 solar masses).
M = 1.989e30 * 10
# Schwarzschild Radius, the radius of the event horizon for a non-rotating black hole.
RS = (2 * G * M) / (c**2)
# A very small number to add to denominators to prevent division-by-zero errors.
EPSILON = 1e-14

# --- Theory-Specific Parameters ---
J_FRAC = 0.5; Q_PARAM = 1e12; Q_UNIFIED = 1e12; ASYMMETRY_PARAM = 1e-4
TORSION_PARAM = 1e-3; OBSERVER_ENERGY = 1e9; LAMBDA_COSMO = 1.11e-52

# --- Tuned Performance Parameters ---
T_SPAN = [0, 100_000] # Reduced simulation time for practical runtimes.
TOLERANCE = 1e-10    # Relaxed tolerance for faster adaptive stepping.

print("="*80)
print(f"DEFINITIVE CPU ORBITAL TEST (FLOAT64) | ALL THEORIES | PARALLEL EXECUTION")
print(f"Utilizing up to {os.cpu_count()} CPU cores for maximum performance.")
print(f"Sim Time: {T_SPAN[1]}, Tolerance: {TOLERANCE:.0e}")
print("="*80)

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# C. PHYSICS ENGINE (NUMPY/SCIPY FOR HIGH PRECISION)
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-

def geodesic_ode(tau, y, model, M_const, C_const, G_const):
    """Defines the ODE system for geodesics using NumPy for high-precision CPU computation."""
    _, r, _, dt, dr, dphi = y
    g_tt, g_rr, g_pp, g_tp = model.get_metric(r, M_const, C_const, G_const)
    g_ttp, g_rrp, g_ppp, g_tpp = model.get_metric_derivatives(r, M_const, C_const, G_const)
    det_inv = g_tt * g_pp + g_tp**2
    if abs(det_inv) < EPSILON: return np.zeros_like(y)

    Gamma_r_tt = -g_ttp/(2*g_rr); Gamma_r_rr = g_rrp/(2*g_rr)
    Gamma_r_pp = -g_ppp/(2*g_rr); Gamma_r_tp = -g_tpp/(2*g_rr)
    d2r = -Gamma_r_tt*dt**2 - Gamma_r_rr*dr**2 - Gamma_r_pp*dphi**2 - 2*Gamma_r_tp*dt*dphi
    d2t = (1/det_inv) * ((g_tp*g_ppp-g_tpp*g_pp)*dr*dphi + (g_tp*g_tpp-g_ttp*g_pp)*dr*dt)
    d2phi = (1/det_inv) * ((g_tpp*g_tt+g_ttp*g_tp)*dr*dphi + (g_ttp*g_pp+g_tp*g_tpp)*dr*dt)
    return [dt, dr, dphi, d2t, d2r, d2phi]

def event_horizon_plunge(tau, y, model, M, C, G):
    """Event function to cleanly stop the solver if the particle crosses the event horizon."""
    return y[1] - RS
event_horizon_plunge.terminal = True; event_horizon_plunge.direction = -1

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# D. GRAVITATIONAL THEORY IMPLEMENTATIONS (Complete Roster)
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-

class GravitationalTheory:
    def __init__(self, name): self.name = name
    def get_metric(self, r, M, C, G): raise NotImplementedError
    def get_metric_derivatives(self, r, M, C, G, h=1e-6):
        p_g = self.get_metric(r+h, M, C, G); m_g = self.get_metric(r-h, M, C, G)
        return tuple([(p - m) / (2 * h) for p, m in zip(p_g, m_g)])

class Schwarzschild(GravitationalTheory):
    def __init__(self): super().__init__("Schwarzschild (GR)")
    def get_metric(self, r, M, C, G): m=1-(2*G*M)/(C**2*r); return -m, 1/(m+EPSILON), r**2, 0.0

class NewtonianLimit(GravitationalTheory):
    def __init__(self): super().__init__("Newtonian Limit")
    def get_metric(self, r, M, C, G): rs=(2*G*M)/(C**2); return -(1-rs/r), 1.0, r**2, 0.0

class Kerr(GravitationalTheory):
    def __init__(self, J_frac): super().__init__(f"Kerr (a*={J_frac:.2f})"); self.a=J_frac*G*M/c**2
    def get_metric(self, r, M, C, G):
        rs,a2,rho2=(2*G*M)/C**2,self.a**2,r**2; delta=rho2-rs*r+a2; g_tt=-(1-rs*r/rho2)
        g_rr=rho2/(delta+EPSILON); g_pp=(r**2+a2)**2-delta*a2; g_tp=-(rs*self.a*r/rho2)*C
        return g_tt/C**2, g_rr, g_pp, g_tp/C

class ReissnerNordstrom(GravitationalTheory):
    def __init__(self, Q): super().__init__(f"Reissner-Nordström (Q={Q:.1e})"); self.Q=Q
    def get_metric(self, r, M, C, G):
        rs,r_q2=(2*G*M)/C**2,(self.Q**2*G)/(4*np.pi*epsilon_0*C**4); m=1-rs/r+r_q2/r**2
        return -m, 1/(m+EPSILON), r**2, 0.0

class EinsteinUnifiedFinal(GravitationalTheory):
    def __init__(self, q): super().__init__(f"Einstein's Unified (q={q:.1e})"); self.q=q
    def get_metric(self, r, M, C, G):
        rs,r_q2=(2*G*M)/C**2,(self.q**2*G)/(4*np.pi*epsilon_0*C**4); m=1-rs/r+r_q2/r**2
        return -m, 1/(m+EPSILON), r**2, 0.0

class EinsteinAsymmetric(GravitationalTheory):
    def __init__(self, alpha): super().__init__(f"Einstein Asymmetric (α={alpha:.1e})"); self.alpha=alpha
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; m=1-rs/r+self.alpha*(rs/r)**2
        return -m, 1/(m+EPSILON), r**2, 0.0

class EinsteinTeleparallel(GravitationalTheory):
    def __init__(self, tau): super().__init__(f"Einstein Teleparallel (τ={tau:.1e})"); self.tau=tau
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; base=1-rs/r; corr=self.tau*(rs/r)**3
        return -base, 1/(base-corr+EPSILON), r**2, 0.0

class EinsteinRegularized(GravitationalTheory):
    def __init__(self): super().__init__("Einstein Regularized Core")
    def get_metric(self, r, M, C, G):
        rs,lp=(2*G*M)/C**2,np.sqrt(G*hbar/C**3); m=1-rs/np.sqrt(r**2+lp**2)
        return -m, 1/(m+EPSILON), r**2, 0.0

class EinsteinFinalEquation(GravitationalTheory):
    def __init__(self, alpha): super().__init__(f"Einstein's Final Eq. (α={alpha:.2f})"); self.alpha=alpha
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; base=1-rs/r; corr=self.alpha*(rs/r)**3; m=base+corr
        return -m,1/(m+EPSILON),r**2,0.0

class Yukawa(GravitationalTheory):
    def __init__(self, lambda_mult): super().__init__(f"Yukawa (λ={lambda_mult:.2f}*RS)"); self.lambda_mult=lambda_mult
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; m=1-(rs/r)*np.exp(-r/(self.lambda_mult*rs))
        return -m, 1/(m+EPSILON), r**2, 0.0

class QuantumCorrected(GravitationalTheory):
    def __init__(self, alpha): super().__init__(f"Quantum Corrected (α={alpha:.2f})"); self.alpha=alpha
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; m=1-rs/r+self.alpha*(rs/r)**3
        return -m, 1/(m+EPSILON), r**2, 0.0

class HigherDimensional(GravitationalTheory):
    def __init__(self,crossover_mult): super().__init__(f"Higher-Dim(cross={crossover_mult:.2f}*RS)"); self.rc=crossover_mult*2*G*M/c**2
    def get_metric(self, r, M, C, G):
        rs,p4d,p5d=(2*G*M)/C**2,rs/r,(self.rc*rs)/r**2; t=1/(1+np.exp(-(r-self.rc)/(self.rc/10)))
        m=1-(t*p4d+(1-t)*p5d); return -m, 1/(m+EPSILON), r**2, 0.0

class LogCorrected(GravitationalTheory):
    def __init__(self, beta): super().__init__(f"Log Corrected (β={beta:.2f})"); self.beta=beta
    def get_metric(self, r, M, C, G):
        rs,sr=(2*G*M)/C**2,np.maximum(r,rs*1.001); lc=self.beta*(rs/sr)*np.log(sr/rs)
        m=1-rs/r+lc; return -m, 1/(m+EPSILON), r**2, 0.0

class VariableG(GravitationalTheory):
    def __init__(self, delta): super().__init__(f"Variable G (δ={delta:.2f})"); self.delta=delta
    def get_metric(self, r, M, C, G):
        rs,G_eff=(2*G*M)/C**2,G*(1+self.delta*rs/r); m=1-(2*G_eff*M)/(C**2*r)
        return -m, 1/(m+EPSILON), r**2, 0.0

class NonLocal(GravitationalTheory):
    def __init__(self): super().__init__("Non-local (Cosmological)")
    def get_metric(self, r, M, C, G):
        rs,ct=(2*G*M)/C**2,(LAMBDA_COSMO*r**2)/3; m=1-rs/r-ct
        return -m, 1/(m+EPSILON), r**2, 0.0

class Fractal(GravitationalTheory):
    def __init__(self, D): super().__init__(f"Fractal (D={D:.2f})"); self.D=D
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; m=1-(rs/r)**(self.D-2.0)
        return -m, 1/(m+EPSILON), r**2, 0.0

class PhaseTransition(GravitationalTheory):
    def __init__(self, crit_mult): super().__init__(f"Phase Transition (crit={crit_mult:.2f}*RS)"); self.rc=crit_mult*2*G*M/c**2
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; n=1-rs/r; co=1-rs/self.rc; m=np.where(r>self.rc, n, co)
        return -m, 1/(m+EPSILON), r**2, 0.0

class Acausal(GravitationalTheory):
    def __init__(self): super().__init__("Acausal (Final State)")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; ht=(hbar*C**3)/(8*np.pi*G*M*k); pt=np.sqrt(hbar*C**5/(G*k**2))
        cf=1-(ht/pt); m=1-(rs*cf)/r; return -m, 1/(m+EPSILON), r**2, 0.0

class Computational(GravitationalTheory):
    def __init__(self): super().__init__("Computational Complexity")
    def get_metric(self, r, M, C, G):
        rs,lp=(2*G*M)/C**2,np.sqrt(G*hbar/C**3); sr=np.maximum(r,lp)
        cf=(rs**2); m=1-cf/(sr*np.log2(sr/lp)); return -m, 1/(m+EPSILON), r**2, 0.0

class Tduality(GravitationalTheory):
    def __init__(self): super().__init__("String T-Duality")
    def get_metric(self, r, M, C, G):
        rs,ls=(2*G*M)/C**2,rs; re=r+ls**2/r; m=1-rs/re
        return -m, 1/(m+EPSILON), r**2, 0.0

class Hydrodynamic(GravitationalTheory):
    def __init__(self): super().__init__("Emergent (Hydrodynamic)")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; vfs=np.minimum((rs*C**2)/r,C**2*0.999999)
        gs=1/(1-vfs/C**2+EPSILON); return -1/gs, gs, r**2, 0.0

class Participatory(GravitationalTheory):
    def __init__(self, obs_energy): super().__init__(f"Participatory (E_obs={obs_energy:.1e})"); self.obs_energy=obs_energy
    def get_metric(self, r, M, C, G):
        rs,Ep=(2*G*M)/C**2,np.sqrt(hbar*C**5/G); cert=1-np.exp(-5*self.obs_energy/Ep)
        g_tt_gr,g_rr_gr=-(1-rs/r),1/(1-rs/r+EPSILON); g_tt_vac,g_rr_vac=-1.0,1.0
        g_tt=cert*g_tt_gr+(1-cert)*g_tt_vac; g_rr=cert*g_rr_gr+(1-cert)*g_rr_vac
        return g_tt, g_rr, r**2, 0.0

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# E. PARALLEL SIMULATION WORKER
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
def run_single_simulation(args):
    """Encapsulates the simulation for a single model to be run in a parallel process."""
    model, ground_truth_final_state, y0, t_span, r0 = args
    
    sol_pred = solve_ivp(
        geodesic_ode, t_span, y0, args=(model, M, c, G),
        method='DOP853', rtol=TOLERANCE, atol=TOLERANCE, events=event_horizon_plunge
    )
    
    final_state_pred = sol_pred.y[:,-1]
    loss = np.sqrt(
        (ground_truth_final_state[1] - final_state_pred[1])**2 +
        (r0 * (ground_truth_final_state[2] - final_state_pred[2]))**2
    )
    if sol_pred.status < 0: loss = 1e50

    # This print happens inside the worker process, so output may be interleaved.
    print(f"  Finished: {model.name:<50} Deviation: {loss:.4e} m")
    return {"Model": model.name, "Loss": loss}

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# F. MAIN SCRIPT EXECUTION
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
if __name__ == "__main__":
    start_time = time.time()
    
    models_to_test = [
        Schwarzschild(), NewtonianLimit(), Acausal(), Kerr(J_frac=J_FRAC),
        ReissnerNordstrom(Q=Q_PARAM), NonLocal(), Computational(), Tduality(),
        Hydrodynamic(), Participatory(obs_energy=OBSERVER_ENERGY), EinsteinUnifiedFinal(q=Q_UNIFIED),
        EinsteinAsymmetric(alpha=ASYMMETRY_PARAM), EinsteinTeleparallel(tau=TORSION_PARAM),
        EinsteinRegularized(),
    ]
    param_sweeps = {
        "Quantum Corrected": (QuantumCorrected, {"alpha": np.linspace(-2.0, 2.0, 10)}),
        "Log Corrected": (LogCorrected, {"beta": np.linspace(-1.5, 1.5, 10)}),
        "Yukawa": (Yukawa, {"lambda_mult": np.logspace(np.log10(1.5), 2, 10)}),
        "Variable G": (VariableG, {"delta": np.linspace(-0.5, 0.5, 10)}),
        "Fractal": (Fractal, {"D": np.linspace(2.95, 3.05, 10)}),
        "Phase Transition": (PhaseTransition, {"crit_mult": np.array([1.5, 2.5, 4.0, 8.0, 16.0])}),
        "Higher-Dimensional": (HigherDimensional, {"crossover_mult": np.array([2.0, 10.0, 20.0, 50.0])}),
        "Einstein's Final Eq.": (EinsteinFinalEquation, {"alpha": np.linspace(-1.0, 1.0, 5)}),
    }
    for name_prefix, (model_class, params) in param_sweeps.items():
        param_key = list(params.keys())[0]
        for val in params[param_key]: models_to_test.append(model_class(**{param_key: val}))
            
    print(f"Initializing robust CPU orbital test for {len(models_to_test)} universes...\n")

    r0 = np.array(4.0*RS, dtype=DTYPE)
    v_tan = np.sqrt(G*M/r0, dtype=DTYPE)
    g_tt_i,_,g_pp_i,_ = Schwarzschild().get_metric(r0, M, c, G)
    dt_dtau_i = np.sqrt(-c**2/(g_tt_i*c**2 + g_pp_i*(v_tan/r0)**2), dtype=DTYPE)
    dphi_dtau_i = (v_tan/r0)*dt_dtau_i
    y0 = np.array([0.0,r0,0.0,dt_dtau_i,0.0,dphi_dtau_i], dtype=DTYPE)
    
    print("Running ground truth simulation with Schwarzschild (GR)...")
    sol_true = solve_ivp(geodesic_ode, T_SPAN, y0, args=(models_to_test[0],M,c,G), method='DOP853', rtol=TOLERANCE, atol=TOLERANCE, events=event_horizon_plunge)
    ground_truth_final_state = sol_true.y[:,-1]
    print(f"GR orbit completed in {sol_true.t[-1]:.1f}s of proper time. Final r={ground_truth_final_state[1]/RS:.4f}*RS\n")
    
    results = [{"Model": models_to_test[0].name, "Loss": 0.0}]
    
    tasks = [(model, ground_truth_final_state, y0, T_SPAN, r0) for model in models_to_test[1:]]
    
    print("--- Submitting All Models to Process Pool for Parallel Execution ---")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_model = {executor.submit(run_single_simulation, task): task for task in tasks}
        for future in as_completed(future_to_model):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                model_name = future_to_model[future][0].name
                print(f"'{model_name}' generated an exception: {exc}")
                results.append({"Model": model_name, "Loss": 1e50})

    print("\n--- DEFINITIVE CPU TEST (FLOAT64): RANKED DEVIATION FROM GENERAL RELATIVITY ---")
    results.sort(key=lambda x: x["Loss"])
    print(f"{'Rank':<5} | {'Model Name':<65} | {'Final Trajectory Deviation (m)':<35}")
    print("="*110)
    for rank, res in enumerate(results, 1):
        loss_str = f"{res['Loss']:.4e}" if res['Loss'] < 1e49 else "Solver Failed"
        print(f"{rank:<5} | {res['Model']:<65} | {loss_str}")
    
    print("="*110)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")