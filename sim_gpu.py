# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# A. LIBRARY IMPORTS AND SETUP
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
import torch
import time
from scipy.constants import G, c, k, hbar, epsilon_0
import numpy as np

# --- Device Setup: The core of the GPU acceleration ---
# Check if the Metal Performance Shaders (MPS) backend is available for Apple Silicon GPUs.
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("✅ PyTorch MPS (Metal) device found. Running on GPU.")
else:
    device = torch.device("cpu")
    print("⚠️ PyTorch MPS device not found. Running on CPU.")

# --- SET DTYPE TO FLOAT32: A necessary compromise for the GPU ---
DTYPE = torch.float32
print(f"⚠️ Using low-precision {DTYPE} for GPU compatibility. Results are for performance testing, not final accuracy.")

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# B. SIMULATION PARAMETERS (as PyTorch Tensors on the GPU)
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
M = torch.tensor(1.989e30 * 10, device=device, dtype=DTYPE)
RS = (2 * G * M) / (c**2)
EPSILON = 1e-7 # Epsilon must be compatible with float32

# --- Theory-Specific Parameters ---
J_FRAC = 0.5; Q_PARAM = 1e12; Q_UNIFIED = 1e12; ASYMMETRY_PARAM = 1e-4
TORSION_PARAM = 1e-3; OBSERVER_ENERGY = 1e9; LAMBDA_COSMO = 1.11e-52

print("="*80)
print(f"PYTORCH-BASED GPU ORBITAL TEST | ALL THEORIES | {M.cpu().numpy()/1.989e30:.1f} M_sol BLACK HOLE")
print("="*80)

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# C. PYTORCH-BASED PHYSICS ENGINE
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-

def get_metric_derivatives_torch(model, r, M_const, C_const, G_const, h=1e-4):
    p_g = model.get_metric(r + h, M_const, C_const, G_const)
    m_g = model.get_metric(r - h, M_const, C_const, G_const)
    return tuple([(p - m) / (2 * h) for p, m in zip(p_g, m_g)])

def geodesic_ode_pytorch(y, model, M_const, C_const, G_const):
    r, dt, dr, dphi = y[1], y[3], y[4], y[5]
    g_tt, g_rr, g_pp, g_tp = model.get_metric(r, M_const, C_const, G_const)
    g_ttp, g_rrp, g_ppp, g_tpp = get_metric_derivatives_torch(model, r, M_const, C_const, G_const)
    det_inv = g_tt * g_pp + g_tp**2
    if torch.abs(det_inv) < EPSILON: return torch.zeros_like(y)
    
    d2r = -1/(2*g_rr) * (g_ttp*dt**2 + g_rrp*dr**2 + g_ppp*dphi**2 + 2*g_tpp*dt*dphi)
    d2t = (1/det_inv) * ((g_tp*g_ppp - g_tpp*g_pp)*dr*dphi + (g_tp*g_tpp - g_ttp*g_pp)*dr*dt)
    d2phi = (1/det_inv) * ((g_tpp*g_tt + g_ttp*g_tp)*dr*dphi + (g_ttp*g_pp + g_tp*g_tpp)*dr*dt)
    
    return torch.stack([dt, dr, dphi, d2t, d2r, d2phi])

def rk4_step(y, func, model, dt):
    k1 = func(y, model, M, c, G)
    k2 = func(y + 0.5*dt*k1, model, M, c, G)
    k3 = func(y + 0.5*dt*k2, model, M, c, G)
    k4 = func(y + dt*k3, model, M, c, G)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# D. GRAVITATIONAL THEORY IMPLEMENTATIONS (PYTORCH FLOAT32 VERSION)
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-

class GravitationalTheory:
    def __init__(self, name): self.name = name
    def get_metric(self, r, M, C, G): raise NotImplementedError

class Schwarzschild(GravitationalTheory):
    def __init__(self): super().__init__("Schwarzschild (GR)")
    def get_metric(self, r, M, C, G):
        m = 1-(2*G*M)/(C**2*r); return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class NewtonianLimit(GravitationalTheory):
    def __init__(self): super().__init__("Newtonian Limit")
    def get_metric(self, r, M, C, G):
        rs= (2*G*M)/(C**2); return -(1-rs/r), torch.tensor(1.0, device=device, dtype=DTYPE), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class ReissnerNordstrom(GravitationalTheory):
    def __init__(self, Q):
        super().__init__(f"Reissner-Nordström (Q={Q:.1e})")
        self.Q = torch.tensor(Q, device=device, dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; r_q2=(self.Q**2*G)/(4*torch.pi*epsilon_0*C**4); m=1-rs/r+r_q2/r**2
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class Kerr(GravitationalTheory):
    def __init__(self, J_frac):
        super().__init__(f"Kerr (a* = {J_frac:.2f})")
        self.a = torch.tensor(J_frac * G * M.item() / c**2, device=device, dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs,a2,rho2=(2*G*M)/C**2,self.a**2,r**2; delta=rho2-rs*r+a2
        g_tt=-(1-rs*r/rho2); g_rr=rho2/(delta+EPSILON); g_pp=(r**2+a2)**2-delta*a2; g_tp=-(rs*self.a*r/rho2)*C
        return g_tt/C**2, g_rr, g_pp, g_tp/C

class EinsteinUnifiedFinal(GravitationalTheory):
    def __init__(self, q): super().__init__(f"Einstein's Unified (q={q:.1e})"); self.q=torch.tensor(q,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; r_q2=(self.q**2*G)/(4*torch.pi*epsilon_0*C**4); m=1-rs/r+r_q2/r**2
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class EinsteinAsymmetric(GravitationalTheory):
    def __init__(self, alpha): super().__init__(f"Einstein Asymmetric (α={alpha:.1e})"); self.alpha=torch.tensor(alpha,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; m=1-rs/r+self.alpha*(rs/r)**2
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class EinsteinTeleparallel(GravitationalTheory):
    def __init__(self, tau): super().__init__(f"Einstein Teleparallel (τ={tau:.1e})"); self.tau=torch.tensor(tau,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; base=1-rs/r; corr=self.tau*(rs/r)**3
        return -base, 1/(base-corr+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class EinsteinRegularized(GravitationalTheory):
    def __init__(self): super().__init__("Einstein Regularized Core")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; lp=torch.sqrt(torch.tensor(G*hbar/C**3, device=device, dtype=DTYPE)); m=1-rs/torch.sqrt(r**2+lp**2)
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class Yukawa(GravitationalTheory):
    def __init__(self, lambda_mult): super().__init__(f"Yukawa (λ={lambda_mult:.2f}*RS)"); self.lambda_mult=torch.tensor(lambda_mult,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; m=1-(rs/r)*torch.exp(-r/(self.lambda_mult*rs))
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class QuantumCorrected(GravitationalTheory):
    def __init__(self, alpha): super().__init__(f"Quantum Corrected (α={alpha:.2f})"); self.alpha=torch.tensor(alpha,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; m=1-rs/r+self.alpha*(rs/r)**3
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class HigherDimensional(GravitationalTheory):
    def __init__(self, crossover_mult): super().__init__(f"Higher-Dim (cross={crossover_mult:.2f}*RS)"); self.rc=torch.tensor(crossover_mult*2*G*M.item()/c**2,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; p4d=rs/r; p5d=(self.rc*rs)/r**2; t=1/(1+torch.exp(-(r-self.rc)/(self.rc/10)))
        m=1-(t*p4d+(1-t)*p5d); return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class LogCorrected(GravitationalTheory):
    def __init__(self, beta): super().__init__(f"Log Corrected (β={beta:.2f})"); self.beta=torch.tensor(beta,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; safe_r=torch.maximum(r,rs*1.001); log_corr=self.beta*(rs/safe_r)*torch.log(safe_r/rs)
        m=1-rs/r+log_corr; return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class VariableG(GravitationalTheory):
    def __init__(self, delta): super().__init__(f"Variable G (δ={delta:.2f})"); self.delta=torch.tensor(delta,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; G_eff=G*(1+self.delta*rs/r); m=1-(2*G_eff*M)/(C**2*r)
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class NonLocal(GravitationalTheory):
    def __init__(self): super().__init__("Non-local (Cosmological)")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; cosmo_term=(LAMBDA_COSMO*r**2)/3; m=1-rs/r-cosmo_term
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class Fractal(GravitationalTheory):
    def __init__(self, D): super().__init__(f"Fractal (D={D:.2f})"); self.D=torch.tensor(D,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; m=1-(rs/r)**(self.D-2.0)
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class PhaseTransition(GravitationalTheory):
    def __init__(self, crit_mult): super().__init__(f"Phase Transition (crit={crit_mult:.2f}*RS)"); self.r_crit=torch.tensor(crit_mult*2*G*M.item()/c**2,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; normal=1-rs/r; condensed=1-rs/self.r_crit
        m=torch.where(r>self.r_crit, normal, condensed)
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class Acausal(GravitationalTheory):
    def __init__(self): super().__init__("Acausal (Final State)")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; hawking_temp=(hbar*C**3)/(8*torch.pi*G*M*k); planck_temp=torch.sqrt(hbar*C**5/(G*k**2))
        corr=1-(hawking_temp/planck_temp); m=1-(rs*corr)/r
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class Computational(GravitationalTheory):
    def __init__(self): super().__init__("Computational Complexity")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; l_p=torch.sqrt(G*hbar/C**3); safe_r=torch.maximum(r,l_p)
        comp_factor=(rs**2); m=1-comp_factor/(safe_r*torch.log2(safe_r/l_p))
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class Tduality(GravitationalTheory):
    def __init__(self): super().__init__("String T-Duality")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; l_s=rs; m=1-rs/(r+l_s**2/r)
        return -m, 1/(m+EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class Hydrodynamic(GravitationalTheory):
    def __init__(self): super().__init__("Emergent (Hydrodynamic)")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; v_flow_sq=torch.minimum((rs*C**2)/r,torch.tensor(C**2*0.999999,device=device,dtype=DTYPE))
        gamma_sq=1/(1-v_flow_sq/C**2+EPSILON); return -1/gamma_sq, gamma_sq, r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class Participatory(GravitationalTheory):
    def __init__(self, obs_energy): super().__init__(f"Participatory (E_obs={obs_energy:.1e})"); self.obs_energy=torch.tensor(obs_energy,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2; E_p=torch.sqrt(hbar*C**5/G); certainty=1-torch.exp(-5*self.obs_energy/E_p)
        g_tt_gr, g_rr_gr=-(1-rs/r),1/(1-rs/r+EPSILON); g_tt_vac, g_rr_vac=-1.0, 1.0
        g_tt=certainty*g_tt_gr+(1-certainty)*g_tt_vac; g_rr=certainty*g_rr_gr+(1-certainty)*g_rr_vac
        return g_tt, g_rr, r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# E. MAIN SIMULATION SCRIPT
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
if __name__ == "__main__":
    total_start_time = time.time()
    
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
        "Higher-Dim": (HigherDimensional, {"crossover_mult": np.array([2.0, 10.0, 20.0, 50.0])})
    }
    for name_prefix, (model_class, params) in param_sweeps.items():
        param_key = list(params.keys())[0]
        for val in params[param_key]:
            models_to_test.append(model_class(**{param_key: val}))
            
    total_models = len(models_to_test)
    print(f"Initializing PyTorch GPU orbital test for {total_models} universes...\n")

    r0 = 4.0 * RS
    v_tan = torch.sqrt(G * M / r0)
    g_tt_i,_,g_pp_i,_=Schwarzschild().get_metric(r0, M, c, G)
    dt_dtau_i=torch.sqrt(-c**2/(g_tt_i*c**2+g_pp_i*(v_tan/r0)**2))
    dphi_dtau_i=(v_tan/r0)*dt_dtau_i
    y0=torch.tensor([0.0,r0.item(),0.0,dt_dtau_i.item(),0.0,dphi_dtau_i.item()],device=device,dtype=DTYPE)

    DT = 2.0; T_MAX = 8_000_000.0; N_STEPS = int(T_MAX / DT)

    print("Running ground truth simulation with Schwarzschild (GR)...")
    y_true = y0.clone()
    step_count = 0
    start_time = time.time()
    
    for step in range(N_STEPS):
        step_count += 1
        y_true = rk4_step(y_true, geodesic_ode_pytorch, models_to_test[0], DT)
        
        # Progress reporting every 100,000 steps
        if step_count % 100000 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step_count:,}/{N_STEPS:,} ({step_count/N_STEPS*100:.1f}%) - "
                  f"r={y_true[1].cpu().item()/RS.cpu().item():.4f}*RS - "
                  f"Elapsed: {elapsed:.1f}s")
        
        if y_true[1] <= RS:
            print(f"  Event horizon reached at step {step_count:,}")
            break
    
    total_time = time.time() - start_time
    ground_truth_final_state = y_true.cpu().numpy()
    print(f"GR orbit completed in {total_time:.2f}s ({step_count:,} steps). "
          f"Final r={ground_truth_final_state[1]/RS.cpu():.4f}*RS, "
          f"φ={np.rad2deg(ground_truth_final_state[2]):.2f}°\n")

    results = []
    print("--- Simulating Orbital Trajectories for All Models (PyTorch GPU) ---")
    for model in models_to_test:
        print(f"Testing: {model.name}...")
        model_start_time = time.time()
        y_pred = y0.clone()
        for i in range(N_STEPS):
            y_pred=rk4_step(y_pred,geodesic_ode_pytorch,model,DT)
            if y_pred[1]<=RS: break
        
        final_state_pred=y_pred.cpu().numpy()
        loss = np.sqrt((ground_truth_final_state[1]-final_state_pred[1])**2 + (r0.item()*(ground_truth_final_state[2]-final_state_pred[2]))**2)
        if not np.isfinite(loss): loss = 1e50
        results.append({"Model": model.name, "Loss": loss})
        print(f"  -> Finished in {time.time()-model_start_time:.2f}s. Deviation: {loss:.4e} m")

    print("\n--- PYTORCH GPU ORBITAL TEST (FLOAT32): RANKED DEVIATION ---")
    results.sort(key=lambda x: x["Loss"])
    print(f"{'Rank':<5} | {'Model Name':<65} | {'Final Trajectory Deviation (m)':<35}")
    print("="*110)
    for rank, res in enumerate(results, 1):
        loss_str = f"{res['Loss']:.4e}" if res['Loss'] < 1e49 else "Solver Failed or Plunged"
        print(f"{rank:<5} | {res['Model']:<65} | {loss_str}")
    
    print("="*110)
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")